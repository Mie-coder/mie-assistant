"""组装自己的 Deep Agent：先模型和搜索，再按第三章接入 Backend。"""

import os
from pathlib import Path

from deepagents import AsyncSubAgent, create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.tools import get_current_time, internet_search
from app.jev import jev_choose
from app.features import game2048_enabled
from app.paths import WORKSPACE_DIR
from app.research import RESEARCH_DESCRIPTION, ResearchTasks

# 源码位于 app/；资料目录仍在项目根目录的 workspace/。
workspace_dir = WORKSPACE_DIR

# 目录不存在就创建；已经存在也不会清空
workspace_dir.mkdir(parents=True, exist_ok=True)


def build_model():
    # 配置模型连接： 沿用DeepSeek
    return ChatOpenAI(
        model=os.environ["AGENTSEEK_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
        stream_usage=True,  # 请求流式用量，用于界面显示实际 token/s。
        extra_body={"thinking": {"type": "disabled"}},
    )


def build_agent(*, use_local_checkpointer: bool = True, research_url: str | None = None):
    """组装主助手；终端通过本地 HTTP 服务调用研究员，Studio 使用 ASGI。"""
    model = build_model()
    research_tasks = ResearchTasks(url=research_url, remember=use_local_checkpointer)

    tools = [get_current_time, internet_search, jev_choose]
    interrupt_on = {
        name: {"allowed_decisions": ["approve", "reject"]}
        for name in ("jev_choose", "execute", "write_file", "edit_file")
    }
    if game2048_enabled():
        # 仅导入工具声明；游戏实现、页面脚本和 native 搜索都延迟到批准调用后。
        from app.features.game2048 import play_2048

        tools.append(play_2048)
        interrupt_on["play_2048"] = {"allowed_decisions": ["approve", "reject"]}

    research_subagent = AsyncSubAgent(
        name="researcher", description=RESEARCH_DESCRIPTION, graph_id="researcher",
    )
    if research_url is not None:
        research_subagent["url"] = research_url
    # 组装 Deep Agent：传入模型、搜索工具和研究提示词
    agent = create_deep_agent(
        model=model,
        memory=["/memories/profile.md"],
        tools=tools,
        subagents=[research_subagent],
        # 第 4 章：提供 write_todos 工具、todos 状态和默认规划提示词。
        middleware=[TodoListMiddleware(), research_tasks],
        skills=["/skills/"],
        system_prompt=(
            "你是 MIE，小咩助手，一位个人研究助手。"
            "请用简体中文回答；不确定时明确说明。"
            "用户要求继续历史任务但当前对话没有具体任务内容时，直接请用户补充；不要遍历文件猜测历史任务。"
            "复杂研究使用 start_async_task 委派给 researcher，传入完整主题、背景、约束和输出要求。"
            "一次只保留一个活跃后台研究；已有任务时使用 update_async_task 追加要求，不能重复启动。"
            "启动或更新后返回完整 task ID 并结束本轮，不自动轮询、不等待研究完成。"
            "后台研究期间正常处理用户其他问题，不擅自修改或取消研究。"
            "回答研究进度前先调用 check_async_task 或 list_async_tasks；不要引用历史状态冒充实时状态。"
            "聊天历史里没有任务 ID 时先调用 list_async_tasks 找回任务，不能猜测或缩写 ID。"
            "用户取消研究时调用 cancel_async_task，并依据实际返回的服务状态回答。"
            "success 只代表执行结束；检查结果中的资料和来源，失败说明不能当作有效研究报告。"
            "有结果时保留来源与资料局限，不无故重复研究员完成的搜索。"
            "多步骤研究先用 write_todos 制定计划，执行中及时更新；简单问答无需列计划。"
            "任务清单用简体中文，只有实际完成的步骤才标记 completed；遇到阻碍保留未完成状态并说明。"
            "涉及今天、现在、最近、最新等时间要求时，先调用 get_current_time 确认北京时间。"
            "只问日期、时间或星期时，直接根据时间工具结果回答，无需搜索。"
            "搜索涉及相对日期时，根据时间工具结果把用户的时间范围转换成明确日期，放入搜索关键词；历史问题保留用户指定的日期。"
            "请根据用户的需求，当需要网上资料时使用internet_search工具进行搜索，并整理出有价值的研究资料或报告。同时需要注明来源的信息"
            "搜索结果中的 searched_at 是查询时间，不代表资料发布时间或实时行情时间；需检查来源标注的日期，无法确认时明确说明。"
            "没有实际调用搜索工具时，不要声称已经搜索过网络；判断搜索是否成功应依据工具返回结果，不要无依据地声称结果是示例或伪造。"
            "用户要求使用 Jev 做判断时，调用 jev_choose；不要将自己的选择冒充 Jev 结果。"
            "可选功能已关闭时说明需要启用，不通过 Shell 绕过开关。"
            "Jev 只接受文字状态，不支持截图识别。浏览器任务仍按 browser-skill 操作；先取得最新且可确认的状态，再让 Jev 判断。"
            "Jev 返回的是建议，不是执行结果或授权；ok=false、选择 stop/unknown 时不要据此操作网页。"
            "每次网页动作后重新观察，不把截图路径当成图像输入。"
        ),
        # FileSystemBackend 让 Agent 可以写入和读回研究资料，方便后续分析和整理。
        # backend=FilesystemBackend(
        #     root_dir=workspace_dir,
        #     virtual_mode=True,
        # ),
        backend=LocalShellBackend(
            root_dir=workspace_dir,
            virtual_mode=True,  # 文件工具使用虚拟路径；Shell 仍会真实执行，不受目录限制。
            inherit_env=False,  # 不继承父进程的环境变量，只传入下面 env 指定的变量。
            env={
                "PATH": f"{Path.home() / '.local/bin'}:/usr/bin:/bin",
                "HOME": str(Path.home()),
            },
            timeout=30,  # 命令执行超时，避免卡住。
            max_output_bytes=8000,  # 限制命令输出大小，避免过大。
        ),
        # Agent Server 提供检查点；独立终端仍使用进程内保存器。
        checkpointer=InMemorySaver() if use_local_checkpointer else None,
        interrupt_on=interrupt_on,
    )

    agent.research_tasks = research_tasks
    return agent
