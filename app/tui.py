"""小咩的终端界面：界面只消费 Agent 事件，不执行研究业务。"""

import asyncio
import json
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from app.approvals import ApprovalRequest, TurnApprovals
from app.research import ASYNC_TOOLS

from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from langgraph.types import Command
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Collapsible, Markdown, Static, TextArea


def welcome_art(*, blink=False, breathe=False) -> Text:
    """用半格字符画小咩；每两行像素合成一行，适合终端的窄长字格。"""
    pixels = [
        "    HHHHHH            HHHHHH    ",
        "   HHhhhhHH          HHhhhhHH   ",
        "  HHhHHHHhHH BBBBBB HHhHHHHhHH  ",
        "  HHhHhhHhHHBBBBBBBBHHhHhhHhHH  ",
        "   HHhhHHHHBBBBBBBBBBHHHHhhHH   ",
        "  HHHHHHBBBBBBBBBBBBBBBBHHHHHH  ",
        "  HHHHBBBBBHHHHHHHHHHBBBBBHHHH  ",
        "     BBBBHHHHHHHHHHHHHHBBBB     ",
        "    BBBHHHHWWWHHHHWWWHHHHBBB    ",
        "    BBBHHHWKKKWHHWKKKWHHHBBB    ",
        "    BBBHHHWKKKWHHWKKKWHHHBBB    ",
        "    BBBHHHHWWWHHHHWWWHHHHBBB    ",
        "    BBBHHHHHHHMHHMHHHHHHHBBB    ",
        "    BBBBHHHHHHHMMHHHHHHHBBBB    ",
        "     BBBBHHHHHHHHHHHHHHBBBB     ",
        "     BBHHHBBBBBBBBBBBBHHHBB     ",
        "     BBHHHHBBBBBBBBBBHHHHBB     ",
        "     BBBHHBBBBBBBBBBBBHHBBO     ",
        "      BBBBBBBBBBBBBBBBBBBO      ",
        "       BBBBBBBBBBBBBBBBBB       ",
        "        HHHHHH    HHHHHH        ",
        "         HHHH      HHHH         ",
    ]
    pixels = [list(row) for row in pixels]
    if blink:
        for row in range(8, 12):
            pixels[row] = ["H" if pixel in "WK" else pixel for pixel in pixels[row]]
        for col in (11, 12, 13, 18, 19, 20):
            pixels[10][col] = "K"
    if breathe:
        # 手掌轻轻下移一个像素，身体轮廓和画布尺寸不变。
        hands = [(row, col) for row in range(15, 18)
                 for col, pixel in enumerate(pixels[row]) if pixel == "H"]
        for row, col in hands:
            pixels[row][col] = "B"
        for row, col in hands:
            pixels[row + 1][col] = "H"
    palette = {"B": "#5278cf", "H": "#f2d5a1", "h": "#b68b56",
               "K": "#171b26", "M": "#805a42", "W": "#fff8e9", "O": "#e99452"}
    art = Text()
    for row in range(0, len(pixels), 2):
        for top, bottom in zip(pixels[row], pixels[row + 1]):
            foreground, background = palette.get(top), palette.get(bottom)
            if foreground and background:
                art.append("▀", style=f"{foreground} on {background}")
            elif foreground:
                art.append("▀", style=foreground)
            elif background:
                art.append("▄", style=background)
            else:
                art.append(" ")
        art.append("\n")
    art.append("你好，我是小咩。", style="bold #dca17b")
    art.append("\n从一个问题开始，一起找答案。")
    return art


class Welcome(Static):
    """只刷新变化的像素帧；离开欢迎页时停止待机动画。"""

    def __init__(self):
        self.frames = {(blink, breathe): welcome_art(blink=blink, breathe=breathe)
                       for blink in (False, True) for breathe in (False, True)}
        self.frame = (False, False)
        self.idle_timer = None
        super().__init__(self.frames[self.frame], classes="welcome", markup=False)

    def on_mount(self):
        self.started = monotonic()
        self.idle_timer = self.set_interval(0.2, self.animate_idle)

    def animate_idle(self):
        elapsed = monotonic() - self.started
        # 约五秒眨一次眼，每次 0.2 秒；双手每 1.2 秒换一次姿态。
        frame = (elapsed % 5 >= 4.8, int(elapsed / 1.2) % 2 == 1)
        if frame != self.frame:
            self.frame = frame
            self.update(self.frames[frame])

    def stop_animation(self):
        if self.idle_timer is not None:
            self.idle_timer.stop()
            self.idle_timer = None

    def on_unmount(self):
        self.stop_animation()


def message_text(message) -> str:
    """只展示正文，不把工具参数或其他内容块当作回答。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")


def error_diagnostic(exc: Exception) -> str:
    """Expose exception types, never request URLs, headers, or raw provider bodies."""
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    names = set(chain)
    if "GraphRecursionError" in names:
        hint = "工具循环达到运行步数上限；应检查重复操作和缺失条件，而不是直接提高上限。"
    elif "gaierror" in names:
        hint = "域名解析失败；检查网络、DNS 或代理地址。"
    elif "ProxyError" in names:
        hint = "代理连接失败；检查当前进程的代理配置。"
    elif names & {"SSLError", "SSLCertVerificationError"}:
        hint = "TLS 连接或证书校验失败；检查证书和代理配置。"
    elif any("Timeout" in name for name in names):
        hint = "请求超时；短句成功不代表长请求也能完成。"
    elif names & {"RemoteProtocolError", "ReadError", "WriteError"}:
        hint = "连接在收发数据时中断；可能来自网络、代理或服务端。"
    elif any("ConnectionError" in name or name == "ConnectError" for name in names):
        hint = "模型连接失败；现有异常不足以确定具体网络原因。"
    else:
        hint = "本轮未完成；请根据异常类型排查。"
    return " → ".join(chain) + "\n" + hint


@dataclass
class UsageStats:
    """按服务端用量统计；模型耗时包含首字等待，不包含工具执行。"""

    starts: dict = field(default_factory=dict)
    output_tokens: int = 0
    model_seconds: float = 0.0
    reported: int = 0
    missing: bool = False

    def start(self, run_id, now):
        self.starts[run_id] = now

    def finish(self, run_id, message, now):
        started = self.starts.pop(run_id, None)
        usage = getattr(message, "usage_metadata", None) or {}
        count = usage.get("output_tokens")
        if started is None or not isinstance(count, int):
            self.missing = True
            return
        self.output_tokens += count
        self.model_seconds += max(0, now - started)
        self.reported += 1

    def summary(self, final=False):
        if not self.reported:
            label = "未提供" if self.missing or final else "等待回报"
            return f"输出 token：{label} · — token/s"
        if self.missing or self.starts:
            return f"已回报 {self.output_tokens} token（部分） · — token/s"
        rate = self.output_tokens / self.model_seconds if self.model_seconds else 0
        return f"输出 {self.output_tokens} token · {rate:.1f} token/s"


class Composer(TextArea):
    """Enter 发送，Ctrl+J 换行；多行粘贴仍按一次输入处理。"""

    class Submitted(Message):
        def __init__(self, text):
            super().__init__()
            self.text = text

    def on_key(self, event: events.Key):
        # 在输入框自己的事件队列里处理 Enter，避免快速输入时抢先读到旧文本。
        if event.key in ("enter", "ctrl+j", "shift+enter"):
            event.stop()
            event.prevent_default()
            if event.key == "enter":
                self.post_message(self.Submitted(self.text))
            else:
                self.insert("\n")


class ToolRecord(Collapsible):
    def __init__(self, name, arguments):
        self.tool_name = name
        self.started = monotonic()
        self.finished = False
        self.label = {"internet_search": "搜索资料", "get_current_time": "获取北京时间",
                      "play_2048": "Jev 玩 2048",
                      "start_async_task": "启动后台研究", "check_async_task": "查询后台研究",
                      "update_async_task": "更新后台研究", "cancel_async_task": "取消后台研究",
                      "list_async_tasks": "列出后台研究"}.get(name, name)
        query = arguments.get("query", "") if isinstance(arguments, dict) else ""
        self.detail = Static("等待工具返回…", markup=False)
        argument_text = "参数：\n" + json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
        extra = []
        if name == "task":
            fields = arguments if isinstance(arguments, dict) else {"input": arguments}
            argument_text = "\n\n".join(
                f"{key}\n{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)}"
                for key, value in fields.items()
            )
            self.raw_detail = Static("Waiting for output…", markup=False)
            extra = [Collapsible(
                Static(json.dumps(arguments, ensure_ascii=False, indent=2, default=str), markup=False),
                self.raw_detail, title="Raw data", collapsed=True,
            )]
        super().__init__(
            Static(argument_text, markup=False),
            self.detail,
            *extra,
            title=f"◌ {self.label}" + (f" · {query}" if query else ""),
            collapsed=True, classes="tool-record",
        )

    def finish(self, output, state="完成"):
        self.finished = True
        self.end_state = state
        failed = getattr(output, "status", None) == "error" or state == "失败"
        payload = getattr(output, "content", output)
        if self.tool_name == "task":
            self.raw_detail.update(repr(output))
        if self.tool_name == "task" or self.tool_name in ASYNC_TOOLS:
            if isinstance(output, Command) and isinstance(output.update, dict):
                messages = output.update.get("messages", [])
                if not isinstance(messages, (list, tuple)):
                    messages = [messages]
                bodies = []
                for message in messages:
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if isinstance(content, list):
                            content = "".join(part.get("text", "") for part in content
                                              if isinstance(part, dict) and part.get("type") == "text")
                    else:
                        content = message_text(message)
                    if isinstance(content, str) and content:
                        bodies.append(content)
                    failed = failed or getattr(message, "status", None) == "error" or (
                        isinstance(message, dict) and message.get("status") == "error")
                if bodies:
                    payload = "\n\n".join(bodies)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                pass
        if self.tool_name == "play_2048" and isinstance(payload, dict) and payload.get("ok") is False:
            failed = True
        summary = "失败" if failed else state
        if state == "完成" and not failed and self.tool_name in ASYNC_TOOLS:
            summary = {
                "start_async_task": "已启动", "update_async_task": "已更新",
                "cancel_async_task": "取消请求已处理", "list_async_tasks": "已查询",
                "check_async_task": "已查询",
            }[self.tool_name]
            if isinstance(payload, dict) and "status" in payload:
                summary = {"pending": "排队中", "running": "运行中", "success": "已完成",
                           "cancelled": "已取消", "interrupted": "已中断", "error": "失败",
                           "timeout": "已超时"}.get(payload["status"], str(payload["status"]))
        if state == "完成" and not failed and isinstance(payload, dict):
            if self.tool_name == "internet_search":
                summary = f"{len(payload.get('results', []))} 条资料"
            elif self.tool_name == "get_current_time":
                summary = f"{payload.get('date', '')} {payload.get('weekday', '')}"
            elif self.tool_name == "play_2048":
                score = (payload.get("final_state") or {}).get("score", 0)
                summary = f"已验证 {payload.get('steps', 0)} 步 · {score} 分"
        icon = "✕" if failed else ("✓" if state == "完成" else "■")
        self.title = f"{icon} {self.label} · {summary} · {monotonic() - self.started:.1f}s"
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if self.tool_name == "task":
            self.detail.update(RichMarkdown("### content\n\n" + text))
        else:
            self.detail.update(text[:8000] + ("\n…详情过长，仅显示前 8000 字符。" if len(text) > 8000 else ""))
        self.set_class(failed, "failed")


class ToolGroup(Collapsible):
    """Collapse a contiguous batch of read-only filesystem operations."""
    NAMES = {"read_file": "Read", "ls": "List", "glob": "Glob", "grep": "Grep"}

    def __init__(self):
        self.records = []
        self.body = Vertical(classes="tool-group-body")
        super().__init__(self.body, title="File operations", collapsed=True, classes="tool-record")

    async def add_record(self, card):
        self.records.append(card)
        await self.body.mount(card)
        self.refresh_summary()

    def refresh_summary(self):
        counts = {}
        for card in self.records:
            label = self.NAMES[card.tool_name]
            counts[label] = counts.get(label, 0) + 1
        pending = sum(not card.finished for card in self.records)
        failed = sum(card.has_class("failed") for card in self.records)
        stopped = sum(getattr(card, "end_state", None) == "停止等待" for card in self.records)
        parts = [f"{name} {count}" for name, count in counts.items()]
        status = f"{pending} running" if pending else f"{len(self.records)} completed"
        if failed or stopped:
            status = f"{pending} running · {failed} failed · {stopped} stopped"
        self.title = " · ".join(parts + [status])


class TaskPlan(Static):
    """展示工具返回的真实清单；本轮结束不等于全部任务完成。"""

    def __init__(self):
        super().__init__("", classes="task-plan", markup=False)
        self.todos = []

    def show_todos(self, todos, interrupted=False):
        self.todos = todos
        completed = sum(item["status"] == "completed" for item in todos)
        text = Text(f"任务计划 · {completed}/{len(todos)} 已完成", style="bold #dca17b")
        if interrupted:
            text.append(" · 本轮未完成（最后状态）", style="#e88f86")
        if not todos:
            text.append("\n暂无任务", style="dim")
        labels = {"pending": ("○ 待开始", "dim"),
                  "in_progress": ("◌ 进行中", "#dca17b"),
                  "completed": ("✓ 已完成", "#88bc94")}
        for item in todos:
            label, style = labels[item["status"]]
            text.append(f"\n{label}  {item['content']}", style=style)
        self.update(text)


class MieApp(App):
    TITLE = "MIE · 小咩助手"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("tab", "toggle_approval_mode", "切换模式", priority=True),
        Binding("ctrl+y", "approve", "批准", priority=True),
        Binding("ctrl+n", "reject", "拒绝", priority=True),
        Binding("escape", "stop", "停止", priority=True),
        Binding("ctrl+q", "quit", "退出", priority=True),
        Binding("pageup", "history_up", show=False, priority=True),
        Binding("pagedown", "history_down", show=False, priority=True),
        Binding("ctrl+end", "latest", show=False, priority=True),
    ]
    CSS = """
    Screen { background: ansi_default; color: ansi_default; }
    #brand { height: 2; padding: 0 2; color: #dca17b; text-style: bold; }
    #chat { height: 1fr; padding: 0 2; scrollbar-size: 1 1; }
    Screen.welcoming #chat { max-height: 14; }
    .welcome { margin: 0 0 1 0; color: $text-muted; height: auto; }
    .user { margin: 1 0; padding: 0 1; border-left: thick #dca17b; height: auto; }
    .answer-label { margin-top: 1; color: #dca17b; height: 1; }
    Markdown { background: transparent; margin: 0 0 1 0; padding: 0; }
    MarkdownH1, MarkdownH2, MarkdownH3 {
        background: transparent; color: ansi_default; text-style: bold;
        border: none; padding: 0; margin: 1 0; content-align: left middle;
    }
    MarkdownTableContent > .header { color: ansi_default; text-style: bold; }
    .tool-record { background: transparent; border: none; padding: 0; margin: 0; }
    .tool-record > CollapsibleTitle { color: $text-muted; padding: 0; }
    .tool-record > Contents { padding: 0 1; border-left: solid #756253; }
    .tool-record Static { height: auto; margin-bottom: 1; }
    .tool-group-body { height: auto; }
    .failed > CollapsibleTitle, .error { color: #e88f86; }
    .notice { color: $text-muted; height: auto; margin: 1 0; }
    .task-plan { height: auto; margin: 1 0; padding: 0 1; border-left: solid #756253; }
    .approval-card { height: auto; margin: 1 0; padding: 0 1; border-left: thick #dca17b; }
    .approval-card Static { height: auto; }
    .approval-actions { height: 3; }
    .approval-actions Button { min-width: 10; width: 1fr; margin-right: 1; }
    #approval-mode { height: auto; color: #88bc94; }
    #approval-mode.automatic { color: #e99452; text-style: bold; }
    #bottom { height: auto; padding: 0 2; }
    #status { height: auto; max-height: 2; color: $text-muted; margin: 0 0 1 0; }
    #input-row { height: auto; border-top: solid #756253; border-bottom: solid #756253; }
    #prompt { width: 3; height: 1; color: #dca17b; margin-top: 1; }
    Composer { height: 3; max-height: 7; width: 1fr; border: none; background: transparent; padding: 1 0; }
    Composer:focus { border: none; }
    #shortcuts { height: auto; max-height: 2; color: $text-muted; padding: 0; }
    """

    def __init__(self, agent, model_name="DeepSeek"):
        super().__init__()
        self.agent = agent
        self.research_tasks = getattr(agent, "research_tasks", None)
        self.model_name = model_name
        self.conversation = []
        self.approvals = TurnApprovals()
        self._approval_future = None
        self._approval_allowed = ()
        self._turn_closing = False
        self.busy = False
        self.turn_worker = None
        self.stats = UsageStats()
        self.started = None
        self.elapsed = 0.0
        self.phase = "就绪"
        self.status_line = Static("就绪", id="status", markup=False)
        self.status_text = "就绪"

    def compose(self) -> ComposeResult:
        yield Static(f"MIE · 小咩助手  /  {self.model_name}", id="brand", markup=False)
        with VerticalScroll(id="chat"):
            yield Welcome()
        with Vertical(id="bottom"):
            yield self.status_line
            yield Static("", id="approval-mode", markup=False)
            with Horizontal(id="input-row"):
                yield Static("❯", id="prompt")
                yield Composer(id="composer", placeholder="输入问题，或 /help 查看帮助", highlight_cursor_line=False)
            yield Static("Enter 发送 · Tab 模式 · Esc 停止回复 · /help", id="shortcuts", markup=False)

    def on_mount(self):
        self.screen.add_class("welcoming")
        self.query_one(Composer).focus()
        self.refresh_approval_mode()
        self.set_interval(0.2, self.refresh_status)

    @property
    def approval_pending(self):
        return self._approval_future is not None and not self._approval_future.done()

    def refresh_approval_mode(self):
        widget = self.query_one("#approval-mode", Static)
        widget.set_class(self.approvals.automatic, "automatic")
        if self.approvals.automatic:
            widget.update("本轮自动 · Shell / 文件操作免逐次审核 · Tab 切回")
        else:
            widget.update("逐步审核 · Tab 切换本轮自动")

    def action_toggle_approval_mode(self):
        # 停止后的收尾阶段不能重新授予自动执行。
        if self.busy and (self._turn_closing or self.phase == "正在停止"):
            return
        self.approvals.toggle()
        self.refresh_approval_mode()
        if self.approvals.automatic and "approve" in self._approval_allowed:
            self.action_approve()

    def _decide(self, decision):
        if self.approval_pending and decision in self._approval_allowed:
            self._approval_future.set_result(decision)

    def action_approve(self):
        self._decide("approve")

    def action_reject(self):
        self._decide("reject")

    @on(Button.Pressed, ".approval-actions Button")
    def decide_approval(self, event):
        self._decide(event.button.name)

    async def review_tool(self, request: ApprovalRequest) -> str:
        title = Static(f"待审批：{request.name}", markup=False)
        arguments = Static(json.dumps(request.arguments, ensure_ascii=False, indent=2), markup=False)
        card = Vertical(title, arguments, classes="approval-card")
        await self.query_one("#chat").mount(card)
        if self.approvals.can_auto_approve(request):
            title.update(f"本轮自动批准：{request.name}")
            return "approve"
        buttons = Horizontal(
            Button("批准 ^Y", name="approve", disabled="approve" not in request.allowed),
            Button("拒绝 ^N", name="reject", disabled="reject" not in request.allowed),
            classes="approval-actions",
        )
        self._approval_future = asyncio.get_running_loop().create_future()
        self._approval_allowed = request.allowed
        choice = None
        try:
            await card.mount(buttons)
            self.phase = "等待审批"
            self.refresh_status()
            # 默认焦点落在拒绝，避免误按 Enter 放行。
            buttons.query(Button).last().focus()
            card.scroll_visible()
            choice = await self._approval_future
            return choice
        finally:
            self._approval_future = None
            self._approval_allowed = ()
            label = {"approve": "已批准", "reject": "已拒绝"}.get(choice, "已停止，未批准")
            title.update(f"{label}：{request.name}")
            await buttons.remove()
            self.query_one(Composer).focus()

    def refresh_status(self):
        if self.started is not None:
            elapsed = monotonic() - self.started if self.busy else self.elapsed
            text = f"{self.phase} · {elapsed:.1f}s · {self.stats.summary(final=not self.busy)}"
        else:
            text = self.phase
        if text != self.status_text:
            self.status_text = text
            self.status_line.update(text)

    @on(TextArea.Changed)
    def resize_composer(self, event):
        event.text_area.styles.height = min(7, max(3, event.text_area.text.count("\n") + 3))

    async def note(self, text, error=False):
        await self.query_one("#chat").mount(Static(text, classes="notice error" if error else "notice", markup=False))

    @on(Composer.Submitted)
    async def submit(self, event):
        question = event.text.strip()
        if not question:
            return
        if question == "/exit":
            self.action_quit()
            return
        if self.busy:
            self.notify("当前回复还在运行。按 Esc 停止后再发送；输入内容已保留。")
            return
        composer = self.query_one(Composer)
        composer.clear()
        for welcome in self.query(Welcome):
            welcome.stop_animation()
        # 欢迎页紧凑排布；开始交互后，聊天区铺满，输入框固定在底部。
        self.screen.remove_class("welcoming")
        chat = self.query_one("#chat", VerticalScroll)
        chat.anchor()
        if question == "/help":
            await self.note(
                "Enter 发送 · Ctrl+J 换行 · Shift+Tab 切换焦点 · Enter 展开工具\n"
                "Tab 切换逐步审核 / 本轮自动；本轮结束、失败或停止后恢复逐步审核。\n"
                "审批时 Ctrl+Y 批准、Ctrl+N 拒绝，也可点击按钮；Tab 可放行当前及本轮后续请求。\n"
                "本轮自动包含 Shell 和文件修改；浏览器扩展自身确认仍独立生效。\n"
                "PageUp / PageDown 翻阅 · Ctrl+End 回到最新 · Ctrl+Q 退出\n"
                "/clear 清空聊天（保留后台研究） · /exit 退出并停止本地研究服务\n"
                "Esc 只停止当前回复，后台研究继续；取消研究请发送明确的取消要求。\n"
                "token/s = 已回报输出 token ÷ 模型调用耗时（含首字等待，不含搜索耗时）。\n"
                "用量通常在每次模型调用结束时返回；失败时保留原始需求和失败说明，不保留未完成工具调用；主动停止的一轮不加入历史。\n"
                "对话只保存在本次进程内，退出后不保留。"
            )
            return
        if question == "/clear":
            await chat.remove_children()
            self.conversation = []
            self.approvals.reset()
            self.refresh_approval_mode()
            self.started = None
            if self.research_tasks and self.research_tasks.tasks:
                await self.note("聊天已清空，后台研究记录仍保留。可发送‘列出后台研究任务’查询或取消。")
            self.phase = "已清空会话"
            self.refresh_status()
            return
        if question.startswith("/"):
            await self.note("未知命令。输入 /help 查看可用命令。")
            return
        self.busy = True
        self._turn_closing = False
        self.started = monotonic()
        self.stats = UsageStats()
        self.phase = "正在生成"
        await chat.query(".welcome").remove()
        await chat.mount(Static(question, classes="user", markup=False))
        self.turn_worker = self.run_worker(self.run_turn(question), exit_on_error=False)

    async def run_turn(self, question):
        chat = self.query_one("#chat", VerticalScroll)
        tools = {}
        groups = {}
        current_group = None
        streams = {}
        result = None
        plan = None
        # 本轮内恢复使用相同 thread_id；新轮带成功历史及失败说明，隔离未完成调用。
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        async def show_plan(todos):
            nonlocal plan
            if plan is None:
                plan = TaskPlan()
                await chat.mount(plan)
            plan.show_todos(todos)

        async def show_text(run_id, text, final=False):
            nonlocal current_group
            if not text:
                return
            if run_id not in streams:
                current_group = None
                widget = Markdown(text if final else "")
                await chat.mount(Static("● 小咩", classes="answer-label"), widget)
                streams[run_id] = (widget, None if final else Markdown.get_stream(widget))
                if final:
                    return
            widget, stream = streams[run_id]
            if final:
                if stream is not None:
                    await stream.stop()
                    streams[run_id] = (widget, None)
                await widget.update(text)
            else:
                await stream.write(text)

        try:
            payload = {"messages": self.conversation + [{"role": "user", "content": question}]}
            if self.research_tasks is not None:
                payload["async_tasks"] = self.research_tasks.snapshot()
            while True:
                result = None
                pauses_by_id = {}
                # v2 事件同时给出文本片段、工具开始/结束和整轮最终状态。
                async for event in self.agent.astream_events(
                    payload, config=config, version="v2",
                ):
                    kind, run_id = event["event"], event["run_id"]
                    data = event.get("data", {})
                    parents = event.get("parent_ids", [])
                    if kind == "on_chat_model_start":
                        self.stats.start(run_id, monotonic())
                    elif kind == "on_chat_model_end":
                        self.stats.finish(run_id, data.get("output"), monotonic())
                    # 子 Agent 的内部输出留在其工具详情里，不混入主回复。
                    if any(parent in tools for parent in parents):
                        continue
                    if kind == "on_chat_model_stream":
                        self.phase = "正在生成"
                        await show_text(run_id, message_text(data.get("chunk")))
                    elif kind == "on_chat_model_end":
                        await show_text(run_id, message_text(data.get("output")), final=True)
                    elif kind == "on_tool_start":
                        card = ToolRecord(event["name"], data.get("input", {}))
                        tools[run_id] = card
                        self.phase = f"正在{card.label}"
                        if card.tool_name in ToolGroup.NAMES:
                            if current_group is None:
                                current_group = ToolGroup()
                                await chat.mount(current_group)
                            groups[run_id] = current_group
                            await current_group.add_record(card)
                        else:
                            current_group = None
                            await chat.mount(card)
                    elif kind == "on_tool_end" and run_id in tools:
                        output = data.get("output")
                        tools[run_id].finish(output)
                        if run_id in groups:
                            groups[run_id].refresh_summary()
                        # 只认成功工具返回的状态更新，不用调用参数预报进度。
                        if tools[run_id].tool_name == "write_todos" and isinstance(output, Command):
                            if isinstance(output.update, dict) and "todos" in output.update:
                                await show_plan(output.update["todos"])
                        self.phase = "正在整理"
                    elif kind == "on_chain_end" and not parents:
                        result = data.get("output")
                    elif kind == "on_chain_stream" and not parents:
                        # astream_events 的暂停来自根图增量，不一定出现在最终 output。
                        chunk = data.get("chunk")
                        if isinstance(chunk, dict):
                            for pause in chunk.get("__interrupt__", ()):
                                pauses_by_id[pause.id] = pause
                if not isinstance(result, dict):
                    raise RuntimeError("本轮未返回完整状态，请重试。")
                for pause in result.get("__interrupt__", ()):
                    pauses_by_id[pause.id] = pause
                pauses = list(pauses_by_id.values())
                if not pauses:
                    break
                payload = await self.approvals.resume(pauses, self.review_tool)
                self.phase = "正在生成"
            if not isinstance(result, dict) or "messages" not in result:
                raise RuntimeError("本轮未返回完整对话，请重试。")
            # 只有完整成功的一轮才进入历史，避免留下没有工具结果的半条调用。
            self.conversation = result["messages"]
            if "todos" in result:
                await show_plan(result["todos"])
            if not streams:
                await show_text("final", message_text(self.conversation[-1]), final=True)
            self.phase = "完成"
        except asyncio.CancelledError:
            self.phase = "已停止"
            await self.note("已停止当前回复。本轮未加入聊天历史；后台研究不会因此取消，可在下一轮查询或取消。已发出的其他工具请求可能仍会完成。")
        except Exception as exc:
            self.phase = "失败"
            # Preserve intent without replaying partial AI tool calls or side effects.
            recovery = (
                "已有后台研究记录仍保留，需先查询实际状态；不要因本轮回复失败而重复启动研究。"
                if self.research_tasks is not None and self.research_tasks.tasks else
                "用户要求重新执行时，应依据上述需求重新开展；这不是从断点恢复。"
            )
            self.conversation = self.conversation + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": (
                    "本轮执行失败，尚未完成上述任务。原始需求已保留；"
                    "本轮中间资料和未完成工具调用未保留，不能声称已经完成研究或核验。"
                    + recovery
                )},
            ]
            await self.note(
                f"本轮失败：{error_diagnostic(exc)}\n"
                "已保留原始需求和失败说明，可要求重新执行；本轮中间资料未保留。",
                error=True,
            )
        finally:
            self._turn_closing = True
            self.approvals.reset()
            self.refresh_approval_mode()
            if plan is not None and self.phase in ("已停止", "失败"):
                plan.show_todos(plan.todos, interrupted=True)
            for _, stream in streams.values():
                if stream is not None:
                    await stream.stop()
            for card in tools.values():
                if not card.finished:
                    state = "停止等待" if self.phase == "已停止" else "失败"
                    card.finish("未收到完整结果。", state=state)
            for group in set(groups.values()):
                group.refresh_summary()
            try:
                checkpointer = getattr(self.agent, "checkpointer", None)
                if checkpointer is not None:
                    await checkpointer.adelete_thread(thread_id)
            finally:
                self.elapsed = monotonic() - self.started
                self.busy = False
                self.refresh_status()

    def action_stop(self):
        self.approvals.reset()
        self.refresh_approval_mode()
        if self.busy and self.turn_worker and not self._turn_closing and self.phase != "正在停止":
            self.phase = "正在停止"
            self.turn_worker.cancel()

    def action_quit(self):
        self.action_stop()
        self.exit()

    def action_history_up(self):
        self.query_one("#chat", VerticalScroll).scroll_page_up()

    def action_history_down(self):
        self.query_one("#chat", VerticalScroll).scroll_page_down()

    def action_latest(self):
        self.query_one("#chat", VerticalScroll).anchor()
