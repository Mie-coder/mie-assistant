"""小咩 2048 插件：一次有边界的授权，真实 Jev 决策，BrowserSkill 执行。"""
import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from time import monotonic
from uuid import uuid4

from app.jev import jev_choose
from app.paths import WORKSPACE_DIR
from .solver import load_solver, recommend

REPORT_DIR = WORKSPACE_DIR / "reports" / "2048"
PAGE_SCRIPT = Path(__file__).with_name("page.js").read_text()
READ_SCRIPT = f"({PAGE_SCRIPT})(null)"
KEYS = {"left": "ArrowLeft", "right": "ArrowRight", "up": "ArrowUp", "down": "ArrowDown"}
# Opaque, one-use continuation handles. Only browsers created by this process
# can be resumed; arbitrary session/tab IDs are never accepted from the model.
_KEPT_BROWSERS = {}

# 经验文件：每轮开始读一次，从中抽取简短策略摘要发给 Jev，不发送整份复盘。
LESSONS_PATH = WORKSPACE_DIR / "memories" / "2048-lessons.md"
# 摘要标题与结尾：说明这是历史经验、只作提示，最终仍以当前棋盘为准。
LESSONS_HEADER = "Lessons from earlier games (apply every move):"
LESSONS_FOOTER = "These lessons are historical hints, not guarantees; the current board still decides."
# 摘要长度上限：只保留前若干条可执行指令，避免把整份复盘发往 Jev。
LESSONS_DIRECTIVE_LIMIT = 6
LESSONS_CHARS_PER_DIRECTIVE = 240
LESSONS_DIRECTIVE_MAX_CHARS = 1600


def _directive_blocks(text):
    """从经验文件里抽出每条的「下次具体如何选择」段落。

    每个条目只取该标题下的**第一段正文**（到第一个空行为止），
    后面的「本局证据 / 对应报告」等复盘噪声不进入摘要。
    只认显式标题行，抽不到就返回空列表，由调用方退化为无经验提示，
    而不是编造经验。
    """
    lines = text.splitlines()
    blocks, collecting, buffer = [], False, []
    for line in lines:
        if line.strip() == "### 下次具体如何选择":
            collecting, buffer = True, []
            continue
        if not collecting:
            continue
        if line.startswith("#"):
            blocks.append(" ".join(buffer).strip())
            collecting, buffer = False, []
            continue
        stripped = line.strip()
        if not stripped:
            # 第一段结束：条目正文到此为止，后面是本局证据。
            if buffer:
                blocks.append(" ".join(buffer).strip())
            collecting, buffer = False, []
            continue
        buffer.append(stripped)
    if collecting and buffer:
        blocks.append(" ".join(buffer).strip())
    return [b for b in blocks if b]


def build_lessons_summary(text):
    """把经验文件正文转成发给 Jev 的简短摘要。

    摘要完全由文件正文派生，保证「报告记录的摘要」与「实际发送的摘要」是同一字符串。
    文件为空或没有可执行条目时返回 None，不返回空壳摘要。
    """
    if not text:
        return None
    directives = _directive_blocks(text)
    if not directives:
        return None
    parts = []
    for index, directive in enumerate(directives[:LESSONS_DIRECTIVE_LIMIT], start=1):
        parts.append(f"({index}) {directive[:LESSONS_CHARS_PER_DIRECTIVE].strip()}")
    summary = " ".join([LESSONS_HEADER, *parts, LESSONS_FOOTER])
    return summary[:LESSONS_DIRECTIVE_MAX_CHARS].strip()


def load_lessons():
    """读取经验文件并派生出简短摘要。

    文件缺失、为空或不可读时返回 None，让调用方在无经验提示下继续，
    不因为缺少经验而中断整局。
    """
    try:
        text = LESSONS_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return build_lessons_summary(text)


class GameError(Exception):
    """只返回固定错误码，不回显网页脚本或凭证。"""


def validate_state(value):
    if not isinstance(value, dict):
        raise GameError("board_unavailable")
    board = value.get("board")
    if (not isinstance(board, list) or len(board) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in board)
            or any(type(n) is not int or n < 0 or (n != 0 and (n < 2 or n & (n-1)))
                   for row in board for n in row)
            or not isinstance(value.get("id"), str) or not value["id"]
            or value.get("state") not in {"fresh", "playing", "gameWon", "gameOver"}
            or any(type(value.get(k)) is not int or value[k] < 0 for k in ("score", "moveCount"))):
        raise GameError("invalid_board")
    return value


def slide(board, direction):
    """只模拟确定性滑动和合并，用于合法动作筛选与结果验收，不代替 Jev 选路。"""
    vertical = direction in {"up", "down"}
    reverse = direction in {"right", "down"}
    lines = [list(row) for row in zip(*board)] if vertical else [row[:] for row in board]
    result, gain = [], 0
    for row in lines:
        numbers = [n for n in (row[::-1] if reverse else row) if n]
        merged, i = [], 0
        while i < len(numbers):
            n = numbers[i]
            if i+1 < len(numbers) and n == numbers[i+1]:
                n *= 2
                gain += n
                i += 1
            merged.append(n)
            i += 1
        merged += [0] * (4-len(merged))
        result.append(merged[::-1] if reverse else merged)
    return ([list(row) for row in zip(*result)] if vertical else result), gain


def legal_directions(board):
    """返回仍能改变棋盘的确定方向；用于判定某个生成结果是否还有活路。"""
    return [direction for direction in KEYS if slide(board, direction)[0] != board]


def spawn_outcomes(board, direction):
    """推演「先滑动、再在某个空位生成 2 或 4」后的全部结果。

    返回 (after_slide, gain, outcomes)；outcomes 是 (spawn, cell, legal_directions) 列表，
    覆盖每个空位的两种生成值。没有空位时 outcomes 为空列表（即已满盘）。
    纯本地确定性推演，不改变真实棋盘，也不预判随机数。
    """
    after_slide, gain = slide(board, direction)
    empties = [(y, x) for y in range(4) for x in range(4) if after_slide[y][x] == 0]
    outcomes = []
    for y, x in empties:
        for spawn in (2, 4):
            candidate = [row[:] for row in after_slide]
            candidate[y][x] = spawn
            outcomes.append((spawn, (y, x), legal_directions(candidate)))
    return after_slide, gain, outcomes


def analyse_directions(board):
    """对每个合法方向判定：是否在「全部 2/4 生成情况」下都不会立即失败。

    返回 (safe, risky)。safe 里的方向，任何空位填 2 或 4 后都仍存在合法动作；
    risky 里的方向，至少有一种生成结果会立即满盘死锁（值是该方向最坏情况说明）。
    满盘（无空位）视为该方向没有可续命的生成结果，归入 risky。
    """
    safe, risky = [], []
    for direction in KEYS:
        after_slide, gain, outcomes = spawn_outcomes(board, direction)
        if after_slide == board:
            continue
        if not outcomes:
            risky.append((direction, "no empty cell; board would be full immediately"))
            continue
        dead = [(spawn, cell) for spawn, cell, legal in outcomes if not legal]
        if dead:
            worst = ", ".join(f"spawn {spawn} at {list(cell)}" for spawn, cell in dead)
            risky.append((direction, f"deadlocks when {worst}"))
        else:
            safe.append(direction)
    return safe, risky


def verify_move(before, after, direction):
    expected, gain = slide(before["board"], direction)
    changes = [(expected[y][x], after["board"][y][x])
               for y in range(4) for x in range(4) if expected[y][x] != after["board"][y][x]]
    if (after["id"] != before["id"] or after["moveCount"] != before["moveCount"]+1
            or after["score"] != before["score"]+gain
            or len(changes) != 1 or changes[0] not in {(0, 2), (0, 4)}):
        raise GameError("move_not_verified")


class Browser2048:
    """复用已连接的 BrowserSkill；只控制自己创建的会话和固定 tab。"""
    def __init__(self):
        self.session = None
        self.tab = None
        self.executable = shutil.which("bsk") or str(Path.home() / ".local/bin/bsk")

    async def command(self, *args):
        env = {k: os.environ[k] for k in ("PATH", "HOME", "BSK_HOME", "TMPDIR", "LANG") if k in os.environ}
        env["BSK_AUTO_START"] = "0"
        try:
            process = await asyncio.create_subprocess_exec(
                self.executable, *args, "--json", env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except OSError as exc:
            raise GameError("browser_unavailable") from exc
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), 25)
        except (TimeoutError, asyncio.CancelledError) as exc:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.communicate()
            if isinstance(exc, TimeoutError):
                raise GameError("browser_timeout") from exc
            raise
        if process.returncode:
            raise GameError("browser_error")
        try:
            data = json.loads(stdout)
        except (ValueError, UnicodeError) as exc:
            raise GameError("browser_invalid_response") from exc
        if not isinstance(data, dict) or data.get("ok") is False:
            raise GameError("browser_error")
        return data

    def scope(self):
        return ("--session", self.session, "--tab-id", str(self.tab))

    async def start(self):
        status = await self.command("status")
        browsers = status.get("browsers", [])
        if len(browsers) != 1:
            raise GameError("connect_one_browser")
        # Do not lose the newly created session id if cancellation arrives at startup.
        opening = asyncio.create_task(self.command("session", "start", "--name", "MIE Jev 2048",
                                                   "--browser", browsers[0]["instance_id"]))
        try:
            created = await asyncio.shield(opening)
        except asyncio.CancelledError:
            created = await opening
            self.session = created.get("session_id")
            raise
        self.session = created.get("session_id")
        if not self.session:
            raise GameError("browser_invalid_response")
        page = await self.command("navigate", "https://play2048.co/", "--session", self.session,
                                  "--wait-until", "domcontentloaded", "--timeout", "20s")
        self.tab = page.get("tab_id")
        if not self.tab or page.get("final_url") != "https://play2048.co/":
            raise GameError("unexpected_page")
        await self.command("observe", *self.scope(), "--max-tokens", "1500")

    async def read(self):
        result = await self.command("evaluate", READ_SCRIPT, *self.scope(), "--timeout", "10s")
        return validate_state(result.get("value"))

    async def move(self, direction, expected):
        action = json.dumps({"direction": direction, "expected": expected}, separators=(",", ":"))
        result = await self.command("evaluate", f"({PAGE_SCRIPT})({action})",
                                    *self.scope(), "--timeout", "5s")
        value = result.get("value")
        if isinstance(value, dict) and value.get("error") in {"board_changed", "input_focused", "invalid_choice"}:
            raise GameError(value["error"])
        return validate_state(value)

    async def new_game(self, expected):
        if expected["state"] != "gameOver" or legal_directions(expected["board"]):
            raise GameError("restart_requires_game_over")
        # The saved store can already be gameOver while the canvas/terminal
        # overlay is still mounting. Wait for a fresh visible control, not a
        # guessed ref or a direct store reset. No clicks occur in this wait.
        for attempt in range(9):
            observed = await self.command("observe", *self.scope(), "--max-tokens", "2000")
            refs = re.findall(r'(@e\d+) button "(?:Play Again|New Game)"', observed.get("text", ""))
            if refs or attempt == 8:
                break
            await asyncio.sleep(.25)
        if len(refs) != 1:
            raise GameError("restart_button_unavailable")
        if await self.read() != expected:
            raise GameError("board_changed")
        action = json.dumps({"restart": True, "expected": expected}, separators=(",", ":"))
        result = await self.command("evaluate", f"({PAGE_SCRIPT})({action})", *self.scope(), "--timeout", "5s")
        value = result.get("value")
        if isinstance(value, dict) and value.get("error"):
            raise GameError("restart_not_verified")
        fresh = validate_state(value)
        tiles = [n for row in fresh["board"] for n in row if n]
        if (fresh["id"] == expected["id"] or fresh["score"] != 0 or fresh["moveCount"] != 0
                or fresh["state"] not in {"fresh", "playing"}
                or len(tiles) != 2 or any(n not in {2, 4} for n in tiles)):
            raise GameError("restart_not_verified")
        return fresh

    async def screenshot(self, path):
        # Only the final screenshot waits for the tile animation, never the move loop.
        await asyncio.sleep(.3)
        await self.command("screenshot", *self.scope(), "--out", str(path))
        return str(path) if path.is_file() else None

    async def close(self):
        if self.session:
            await self.command("session", "stop", self.session)
            self.session = None


async def _choose(current, choices, lessons=None):
    state = json.dumps({"game": "2048", "board_rows_top_to_bottom": current["board"],
                        "empty_cell": 0, "score": current["score"]}, separators=(",", ":"))
    question = ("Which legal move best advances toward 2048? Prefer empty space, mergeable tiles, "
                "and keeping the largest tiles together near a corner. Each choice includes the exact "
                "board AFTER sliding but BEFORE a random 2 or 4 spawns. Each choice states its "
                "immediate-death risk; when all moves are risky, compare their chances rather than "
                "assuming they are equally bad. "
                "Choose stop only if uncertain.")
    # 经验摘要与棋盘一起发往 Jev；缺省时退化为原来的问题文本。
    if lessons:
        question = f"{question}\n{lessons}"
    # Cancellation stops the coroutine before it can issue any later keyboard action.
    # An already-sent HTTP request may still finish in this thread (jev_choose timeout: 30s).
    return await asyncio.to_thread(jev_choose, state, question, choices)


def reached_goal(current):
    return any(2048 in row for row in current["board"])


async def run_game(max_steps=50, max_seconds=120, search_assist=False,
                   new_game=False, keep_page_open=False, continuation_id=None):
    if type(max_steps) is not int or not 1 <= max_steps <= 200 or type(max_seconds) is not int or not 1 <= max_seconds <= 300:
        return {"ok": False, "reason": "invalid_limits", "message": "步数须为 1–200 的整数，时间须为 1–300 秒的整数。"}
    if any(type(value) is not bool for value in (search_assist, new_game, keep_page_open)):
        return {"ok": False, "reason": "invalid_options"}
    if continuation_id is not None and (not isinstance(continuation_id, str) or continuation_id not in _KEPT_BROWSERS):
        return {"ok": False, "reason": "invalid_continuation", "message": "保留会话不存在；重启进程后须重新读取网站保存的棋盘。"}
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        return {"ok": False, "reason": "not_configured", "message": "请设置 TYPESAFE_API_KEY 并重启小咩。"}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"game-{uuid4().hex}.json"
    kept = _KEPT_BROWSERS.pop(continuation_id) if continuation_id else None
    browser = kept[0] if kept else Browser2048()
    started = monotonic()
    report = {"ok": False, "reason": "running", "steps": 0, "max_steps": max_steps,
              "max_seconds": max_seconds, "decisions": [], "report_path": str(report_path),
              "lessons_path": str(LESSONS_PATH), "lessons_loaded": False,
              "lessons_summary": None, "lessons_sent_to_jev": False,
              "search_assist": search_assist, "overrides": 0, "new_game_requested": new_game,
              "page_preserved": False}
    current = None

    def save():
        report["elapsed_seconds"] = round(monotonic()-started, 3)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    try:
        async with asyncio.timeout(max_seconds):
            # 每轮开始读一次经验文件；读不到就无经验提示继续，不中断整局。
            lessons = load_lessons()
            report["lessons_loaded"] = lessons is not None
            # 报告记录的摘要就是实际发给 Jev 的那个字符串，二者同源，避免记录与请求不一致。
            report["lessons_summary"] = lessons
            save()
            if search_assist:
                try:
                    await asyncio.to_thread(load_solver)
                except (RuntimeError, OSError):
                    raise GameError("solver_unavailable")
            if not kept:
                await browser.start()
            read_started = monotonic()
            current = validate_state(await browser.read())
            if kept and current != kept[1]:
                raise GameError("board_changed")
            if new_game:
                report["previous_game"] = current
                current = await browser.new_game(current)
                report["new_game_verified"] = True
            read_seconds = monotonic()-read_started
            report["initial_state"] = current
            for step in range(1, max_steps+1):
                if reached_goal(current):
                    report["reason"] = "game_won"
                    break
                if current["state"] == "gameWon":
                    raise GameError("win_not_verified")
                if current["state"] == "gameOver":
                    if legal_directions(current["board"]):
                        raise GameError("game_over_not_verified")
                    report["reason"] = "game_over"
                    break
                # 先做本地推演：只把「任何 2/4 生成情况都不会立即失败」的方向给 Jev。
                safe, risky = analyse_directions(current["board"])
                # 安全方向优先；一个都没有时，如实退回全部「能合并/移动」的方向 + stop。
                offered = safe if safe else [direction for direction, _ in risky]
                choices = {}
                for direction in offered:
                    after_slide, gain, _ = spawn_outcomes(current["board"], direction)
                    choices[direction] = (f"Move {direction}; score gain {gain}; empty cells "
                                          f"{sum(row.count(0) for row in after_slide)}; board {after_slide}; "
                                          f"risk: {dict(risky).get(direction, 'no immediate deadlock for any spawn')}")
                if not choices:
                    # 没有任何方向能改变棋盘：真实终局，不含随机因素。
                    report["reason"] = "game_over"
                    break
                choices["stop"] = "Stop playing if no reliable decision can be made."
                decision = {"step": step, "before": current, "choices": choices,
                            "safe_directions": list(safe),
                            "risky_directions": [{"direction": d, "why": why} for d, why in risky],
                            "filtered_to_safe": bool(safe),
                            "read_seconds": round(read_seconds, 3), "verified": False}
                report["decisions"].append(decision)
                save()
                called = monotonic()
                search_task = asyncio.create_task(asyncio.to_thread(recommend, current["board"], offered)) if search_assist else None
                try:
                    result = await _choose(current, choices, lessons)
                    decision.update(result=result, jev_seconds=round(monotonic()-called, 3))
                    if search_task and result.get("ok") and result.get("choice") != "stop":
                        try:
                            decision["search"] = await search_task
                        except Exception as exc:
                            raise GameError("solver_error") from exc
                finally:
                    if search_task:
                        search_task.cancel()
                        await asyncio.gather(search_task, return_exceptions=True)
                decision["decision_seconds"] = round(monotonic()-called, 3)
                decision["lessons_sent"] = bool(lessons)
                if lessons:
                    report["lessons_sent_to_jev"] = True
                save()
                if not result.get("ok"):
                    raise GameError("jev_error")
                direction = result.get("choice")
                if direction == "stop":
                    report["reason"] = "jev_stop"
                    break
                if direction not in choices:
                    raise GameError("invalid_choice")
                decision["jev_direction"] = direction
                if search_assist:
                    search = decision["search"]
                    if search["direction"] not in offered:
                        raise GameError("solver_invalid_choice")
                    # Preserve Jev on an exact score tie. The core's heuristic score
                    # is NOT a probability or a calibrated confidence estimate.
                    if search["scores"][search["direction"]] > search["scores"][direction]:
                        direction = search["direction"]
                decision["executed_direction"] = direction
                decision["overridden"] = direction != result["choice"]
                report["overrides"] += int(decision["overridden"])
                action_started = monotonic()
                decision["action_attempted"] = True
                save()
                # Guard, keyboard event, and readback share one browser operation.
                after = validate_state(await browser.move(direction, current))
                decision["after"] = after
                before, current = current, after
                verify_move(before, after, direction)
                decision.update(verified=True, action_verify_seconds=round(monotonic()-action_started, 3))
                report["steps"] += 1
                read_seconds = 0  # Subsequent reads are included in action_verify_seconds.
                save()
            else:
                report["reason"] = "step_limit"
            if reached_goal(current):
                report["reason"] = "game_won"
            elif current["state"] == "gameWon":
                raise GameError("win_not_verified")
            elif current["state"] == "gameOver":
                if legal_directions(current["board"]):
                    raise GameError("game_over_not_verified")
                report["reason"] = "game_over"
    except asyncio.CancelledError:
        report["reason"] = "cancelled"
        raise
    except TimeoutError:
        report["reason"] = "time_limit"
    except GameError as exc:
        report["reason"] = str(exc)
    except Exception:
        report["reason"] = "unexpected_error"
    finally:
        report["final_state"] = current
        report["ok"] = report["reason"] in {"step_limit", "time_limit", "game_won", "game_over", "jev_stop"}
        save()
        try:
            if report["ok"] and current:
                try:
                    report["screenshot_path"] = await browser.screenshot(report_path.with_suffix(".png"))
                except (GameError, TimeoutError):
                    report["screenshot_path"] = None
        finally:
            try:
                if keep_page_open and getattr(browser, "session", None):
                    handle = uuid4().hex
                    _KEPT_BROWSERS[handle] = (browser, current)
                    report.update(page_preserved=True, continuation_id=handle,
                                  browser_session=browser.session, browser_tab_id=browser.tab,
                                  page_location="agent_window")
                else:
                    await browser.close()
            except (GameError, TimeoutError):
                report.update(ok=False, cleanup_error="session_close_failed")
            save()
    summary = {k: v for k, v in report.items() if k not in {"decisions", "initial_state"}}
    summary["jev_calls"] = len(report["decisions"])
    summary["moves"] = [{"direction": d.get("executed_direction"),
                          "jev_direction": d.get("result", {}).get("choice"),
                          "confidence": d.get("result", {}).get("confidence"),
                          "jev_seconds": d.get("jev_seconds"), "verified": d["verified"]}
                         for d in report["decisions"][-10:]]
    return summary
