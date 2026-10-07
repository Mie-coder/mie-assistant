"""真实本地 Agent Server + 官方异步工具；模型和搜索使用离线替身。"""

import asyncio
import json
import unittest

from deepagents import AsyncSubAgent, create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.research import ResearchTasks
from app.research_server import LocalResearchServer
from app.tui import Composer, MieApp
from tests.test_tui import ScriptedModel


class ResearchModel(ScriptedModel):
    """按用户指令选择真实工具；不伪造服务端状态。"""

    tasks: object = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            answer = AIMessage(content=str(last.content))
        else:
            text = last.content
            task_id = next(reversed(self.tasks.tasks), "unknown")
            if text.startswith("启动"):
                name, args = "start_async_task", {"subagent_type": "researcher", "description": text}
            elif text.startswith("更新"):
                name, args = "update_async_task", {"task_id": task_id, "message": text}
            elif text == "查询":
                name, args = "check_async_task", {"task_id": task_id}
            elif text == "取消":
                name, args = "cancel_async_task", {"task_id": task_id}
            elif text == "列表":
                name, args = "list_async_tasks", {}
            else:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="普通聊天正常"))])
            answer = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-" + str(len(messages))}])
        return ChatResult(generations=[ChatGeneration(message=answer)])


def make_graph(url):
    tasks = ResearchTasks(url=url)
    model = ResearchModel(responses=[], tasks=tasks)
    graph = create_deep_agent(
        model=model,
        subagents=[AsyncSubAgent(name="researcher", description="测试研究员", graph_id="researcher", url=url)],
        middleware=[tasks], checkpointer=InMemorySaver(),
    )
    graph.research_tasks = tasks
    return graph


class BackgroundResearchTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = LocalResearchServer(graph_path="tests/research_fixture.py:graph", env_file=None)
        cls.server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    async def send(self, app, pilot, text):
        app.query_one(Composer).load_text(text)
        await pilot.press("enter")
        for _ in range(300):
            if not app.busy:
                return
            await asyncio.sleep(.02)
        self.fail("主 Agent 没有结束本轮")

    async def test_background_chat_update_result_and_single_slot(self):
        graph = make_graph(self.server.url)
        app = MieApp(graph)
        async with app.run_test() as pilot:
            await self.send(app, pilot, "启动资料研究")
            task_id, initial = next(iter(graph.research_tasks.tasks.items()))
            old_run = initial["run_id"]
            run = await graph.research_tasks.client.runs.get(task_id, old_run)
            self.assertIn(run["status"], {"pending", "running"})
            await self.send(app, pilot, "你好")
            self.assertEqual(app.phase, "完成")
            await self.send(app, pilot, "启动第二个任务")
            self.assertEqual(len(graph.research_tasks.tasks), 1)
            self.assertIn("已有", str(app.conversation[-1].content))
            await self.send(app, pilot, "更新：只看官方文档")
            updated = graph.research_tasks.tasks[task_id]
            self.assertNotEqual(updated["run_id"], old_run)
            self.assertEqual(list(graph.research_tasks.tasks), [task_id])
            await graph.research_tasks.client.runs.join(task_id, updated["run_id"])
            await self.send(app, pilot, "查询")
            result = json.loads(app.conversation[-1].content)["result"]
            self.assertIn("启动资料研究", result)
            self.assertIn("只看官方文档", result)
            self.assertIn("https://docs.langchain.com/", result)
            self.assertEqual(graph.research_tasks.tasks[task_id]["status"], "success")
            self.assertEqual(graph.checkpointer.storage, {})

    async def test_cancel_is_confirmed_and_clear_keeps_task_access(self):
        graph = make_graph(self.server.url)
        app = MieApp(graph)
        async with app.run_test() as pilot:
            await self.send(app, pilot, "启动取消测试")
            task_id = next(iter(graph.research_tasks.tasks))
            await self.send(app, pilot, "/clear")
            self.assertEqual(app.conversation, [])
            await self.send(app, pilot, "列表")
            self.assertIn(task_id, str(app.conversation[-1].content))
            await self.send(app, pilot, "取消")
            task = graph.research_tasks.tasks[task_id]
            run = await graph.research_tasks.client.runs.get(task_id, task["run_id"])
            self.assertIn(run["status"], {"interrupted", "cancelled"})
            self.assertEqual(task["status"], "cancelled")
            self.assertIn("server_status", str(app.conversation[-1].content))

    async def test_stop_or_failure_after_launch_does_not_lose_background_task(self):
        for mode in ("stop", "failure"):
            with self.subTest(mode=mode):
                graph = make_graph(self.server.url)
                ready = asyncio.Event()

                class InterruptedReply:
                    research_tasks = graph.research_tasks
                    checkpointer = graph.checkpointer
                    first = True

                    async def astream_events(self, *args, **kwargs):
                        async for event in graph.astream_events(*args, **kwargs):
                            yield event
                            if self.first and event["event"] == "on_tool_end" and event["name"] == "start_async_task":
                                self.first = False
                                ready.set()
                                if mode == "failure":
                                    raise RuntimeError("测试：启动任务后主回复失败")
                                await asyncio.Event().wait()

                app = MieApp(InterruptedReply())
                async with app.run_test() as pilot:
                    app.query_one(Composer).load_text("启动停止回复测试")
                    await pilot.press("enter")
                    await asyncio.wait_for(ready.wait(), 3)
                    if mode == "stop":
                        await pilot.press("escape")
                    for _ in range(100):
                        if not app.busy:
                            break
                        await asyncio.sleep(.02)
                    self.assertFalse(app.busy)
                    self.assertEqual(app.phase, "已停止" if mode == "stop" else "失败")
                    task_id, task = next(iter(graph.research_tasks.tasks.items()))
                    run = await graph.research_tasks.client.runs.get(task_id, task["run_id"])
                    self.assertIn(run["status"], {"pending", "running"})
                    await self.send(app, pilot, "列表")
                    self.assertIn(task_id, str(app.conversation[-1].content))
                    await self.send(app, pilot, "取消")
                    self.assertEqual(graph.research_tasks.tasks[task_id]["status"], "cancelled")

    async def test_parallel_start_calls_launch_only_one_research(self):
        tasks = ResearchTasks(url=self.server.url)
        model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[
            {"name": "start_async_task", "id": "a", "args": {"subagent_type": "researcher", "description": "任务 A"}},
            {"name": "start_async_task", "id": "b", "args": {"subagent_type": "researcher", "description": "任务 B"}},
        ]), AIMessage(content="已处理启动请求")])
        graph = create_deep_agent(
            model=model,
            subagents=[AsyncSubAgent(name="researcher", description="研究", graph_id="researcher", url=self.server.url)],
            middleware=[tasks],
        )
        result = await graph.ainvoke({"messages": [{"role": "user", "content": "研究"}]})
        self.assertEqual(len(tasks.tasks), 1)
        denied = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
        self.assertEqual(len(denied), 1)
        self.assertIn("已有", denied[0].content)
        task_id, task = next(iter(tasks.tasks.items()))
        await tasks.client.runs.cancel(task_id, task["run_id"], wait=True)


if __name__ == "__main__":
    unittest.main()
