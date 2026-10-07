"""官方异步工具的应用约束：单任务、跨轮登记、取消终态确认。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Overwrite
from langgraph_sdk import get_client


RESEARCH_DESCRIPTION = (
    "用于需要多次搜索官方资料、交叉核对来源并综合成摘要的后台研究任务，"
    "例如技术方案对比、指定时间范围的更新调研。"
    "单个事实、日期或简单定义查询由主 Agent 直接处理。"
)

RESEARCH_PROMPT = (
    "你是小咩助手的研究员，请用简体中文完成委派的研究任务。\n"
    "1. 结合本会话已有研究主题和最新追加要求开展研究；追加要求不会替代原主题。"
    "必要的背景、约束或日期范围缺失时说明缺口，不自行猜测。\n"
    "2. 使用 internet_search 搜索官方资料，核对来源身份和发布日期；"
    "不能将搜索时间当作发布日期。\n"
    "3. 只报告有资料支持的发现，不编造事实或来源链接。\n"
    "4. 每条关键发现分别返回：结论、对应官方页面标题和 URL、资料局限。"
    "未确认的信息或适用范围限制要明确说明；不编造局限。\n"
    "5. 返回精炼摘要，正文尽量控制在 500 字以内，来源链接单独列出。\n"
    "6. 搜索失败或没有找到符合条件的资料时，如实说明，不声称研究已成功完成。"
)

ASYNC_TOOLS = {
    "start_async_task", "check_async_task", "update_async_task",
    "cancel_async_task", "list_async_tasks",
}
TERMINAL = {"success", "error", "timeout", "interrupted", "cancelled"}


class InterruptedResearchHistory(AgentMiddleware):
    """更新中断工具时，只移除未配齐结果的调用组，不伪造成功结果。"""

    def before_model(self, state, runtime):
        messages = state["messages"]
        clean = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if isinstance(message, AIMessage) and message.tool_calls:
                end = index + 1
                while end < len(messages) and isinstance(messages[end], ToolMessage):
                    end += 1
                expected = {call["id"] for call in message.tool_calls}
                returned = [item.tool_call_id for item in messages[index + 1:end]]
                if len(returned) == len(expected) and set(returned) == expected:
                    clean.extend(messages[index:end])
                index = end
                continue
            if not isinstance(message, ToolMessage):
                clean.append(message)
            index += 1
        if len(clean) != len(messages):
            return {"messages": Overwrite(clean)}
        return None


class ResearchTasks(AgentMiddleware):
    """不重写五个工具；只包装官方实现并保留任务操作的真实回执。

    终端每轮会清理临时图检查点，但任务账本独立于消息历史。
    Studio 使用服务端持久化状态，不共享不同主 thread 的账本。
    """

    def __init__(self, url=None, *, remember=True):
        self.client = get_client(url=url, timeout=10)
        self.remember = remember
        self.tasks = {}
        self.lock = asyncio.Lock()

    def snapshot(self):
        return deepcopy(self.tasks)

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] in ASYNC_TOOLS:
            return self._error(request, "后台研究工具需要异步调用，请使用小咩终端入口或 Agent Server。")
        return handler(request)

    @staticmethod
    def _error(request, message):
        return ToolMessage(
            message, name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"], status="error",
        )

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in ASYNC_TOOLS:
            return await handler(request)
        # 只保护短时任务管理请求，不等待整个研究。Esc 不能丢掉已启动任务的 ID。
        operation = asyncio.create_task(self._call(request, handler))
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            await operation
            raise

    async def _call(self, request, handler):
        async with self.lock:
            tasks = dict(request.state.get("async_tasks") or {})
            if self.remember:
                tasks.update(self.snapshot())
            state = {**request.state, "async_tasks": tasks}
            request = replace(request, state=state, runtime=replace(request.runtime, state=state))
            name = request.tool_call["name"]
            target = request.tool_call["args"].get("task_id")
            instructions = []
            if name == "start_async_task":
                instructions = [request.tool_call["args"].get("description", "")]
            elif name == "update_async_task" and target in tasks:
                instructions = list(tasks[target].get("instructions", []))
                instructions.append(request.tool_call["args"].get("message", ""))
                # 旧 run 可能尚未写入第一个检查点。显式带上原始主题和历次追加要求，
                # 不能假设 interrupt 后服务端一定已经保存了那条输入消息。
                content = "研究任务与历次补充（按时间顺序，冲突时以最新要求为准）：\n" + "\n".join(instructions)
                request = request.override(tool_call={
                    **request.tool_call,
                    "args": {**request.tool_call["args"], "message": content},
                })

            try:
                if name in {"start_async_task", "update_async_task"}:
                    for task_id, task in tasks.items():
                        if task_id == target or task["status"] in TERMINAL:
                            continue
                        run = await self.client.runs.get(task_id, task["run_id"])
                        if run["status"] not in TERMINAL:
                            return self._error(request, f"已有后台研究任务 {task_id}。请查询、更新或取消它；当前只支持一个活跃任务。")
                result = await handler(request)
            except Exception as exc:
                return self._error(request, f"后台任务操作失败（{type(exc).__name__}），请检查本地研究服务后重试。")

            if isinstance(result, Command) and isinstance(result.update, dict):
                changes = result.update.get("async_tasks", {})
                for task_id, task in changes.items():
                    task["instructions"] = instructions or tasks.get(task_id, {}).get("instructions", [])
                if name == "cancel_async_task" and target in changes:
                    # 官方工具先记 cancelled；这里再读服务端，不能拿本地标记冒充终态。
                    task = changes[target]
                    try:
                        async with asyncio.timeout(10):
                            while True:
                                run = await self.client.runs.get(target, task["run_id"])
                                if run["status"] in TERMINAL:
                                    break
                                await asyncio.sleep(.1)
                        task["status"] = "cancelled" if run["status"] in {"cancelled", "interrupted"} else run["status"]
                        result.update["messages"] = [ToolMessage(
                            json.dumps({"task_id": target, "status": task["status"], "server_status": run["status"]}, ensure_ascii=False),
                            name=name, tool_call_id=request.tool_call["id"],
                        )]
                    except Exception:
                        task["status"] = "running"
                        result.update["messages"] = [self._error(request, "取消请求已发送，但服务端终态尚未确认。请稍后查询，不要声称任务已停止。")]
                if self.remember:
                    self.tasks.update(deepcopy(changes))
            return result
