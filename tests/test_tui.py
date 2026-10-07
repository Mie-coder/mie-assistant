"""终端界面的离线回归：不读取 .env，不调用模型或搜索服务。"""

import asyncio
import json
import unittest
from unittest.mock import patch

from app.agent import build_agent

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from deepagents import create_deep_agent
from app.tools import get_current_time
from textual import events
from textual.widgets import Collapsible, Markdown, Static
from textual.containers import VerticalScroll

from app.tui import Composer, MieApp, UsageStats, Welcome


class DemoAgent:
    def __init__(self, mode="success"):
        self.mode = mode
        self.inputs = []
        self.waiting = asyncio.Event()

    async def astream_events(self, data, *, config=None, version):
        self.inputs.append(data)
        yield {"event": "on_tool_start", "name": "internet_search", "run_id": "tool", "parent_ids": ["root"], "data": {"input": {"query": "中文资料"}}}
        if self.mode == "waiting":
            self.waiting.set()
            await asyncio.Event().wait()
        if self.mode == "error":
            raise RuntimeError("搜索服务暂时不可用")
        yield {"event": "on_tool_end", "name": "internet_search", "run_id": "tool", "parent_ids": ["root"], "data": {"output": ToolMessage(content='{"results": [{"title": "资料", "url": "https://example.com"}]}', tool_call_id="call")}}
        yield {"event": "on_chat_model_start", "run_id": "model", "parent_ids": ["root"], "data": {}}
        text = "## 中文回答\n\n| 项目 | 内容 |\n|---|---|\n| 来源 | 官方资料 |\n\n```python\nprint('小咩')\n```"
        for chunk in (text[:10], text[10:]):
            yield {"event": "on_chat_model_stream", "run_id": "model", "parent_ids": ["root"], "data": {"chunk": AIMessageChunk(content=chunk)}}
        yield {"event": "on_chat_model_end", "run_id": "model", "parent_ids": ["root"], "data": {"output": AIMessage(content=text)}}
        yield {"event": "on_chain_end", "run_id": "root", "parent_ids": [], "data": {"output": {"messages": data["messages"] + [AIMessage(content=text)]}}}


class ScriptedModel(FakeMessagesListChatModel):
    """用真正的 Deep Agents 图跑事件，但用本地预设回复替代网络模型。"""

    def bind_tools(self, tools, **kwargs):
        return self


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_planning_real_tool_updates_and_unfinished_final(self):
        initial = [{"content": "查资料 [中文]", "status": "in_progress"},
                   {"content": "写报告", "status": "pending"}]
        updated = [{"content": "查资料 [中文]", "status": "completed"},
                   {"content": "写报告", "status": "in_progress"}]
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{"id": "p1", "name": "write_todos", "args": {"todos": initial}}]),
            AIMessage(content="", tool_calls=[{"id": "p2", "name": "write_todos", "args": {"todos": updated}}]),
            AIMessage(content="资料已整理，报告还未完成。"),
        ])
        with patch.dict("os.environ", {"AGENTSEEK_MODEL": "offline", "OPENAI_API_KEY": "offline", "OPENAI_API_BASE": "https://example.invalid"}), patch("app.agent.ChatOpenAI", return_value=model):
            graph = build_agent()
        ready, proceed = asyncio.Event(), asyncio.Event()

        class PausedGraph:
            async def astream_events(self, data, *, config=None, version):
                async for event in graph.astream_events(data, config=config, version=version):
                    yield event
                    if event["event"] == "on_tool_end" and event["name"] == "write_todos" and not ready.is_set():
                        ready.set()
                        await proceed.wait()

        app = MieApp(PausedGraph())
        async with app.run_test(size=(48, 24)) as pilot:
            await self.send(app, pilot, "研究并写报告")
            await asyncio.wait_for(ready.wait(), 3)
            panel = app.query_one(".task-plan", Static)
            self.assertIn("待开始", str(panel.render()))
            self.assertIn("查资料 [中文]", str(panel.render()))
            proceed.set()
            await self.settle(app)
            self.assertEqual(app.phase, "完成")
            self.assertEqual(len(app.query(".task-plan")), 1)
            self.assertIn("1/2", str(panel.render()))
            self.assertIn("进行中", str(panel.render()))
            self.assertLessEqual(app.query_one(Composer).region.bottom, 24)
            await self.send(app, pilot, "/clear")
            self.assertFalse(app.query(".task-plan"))

    async def test_plan_survives_stop_or_error_without_fake_completion(self):
        from langgraph.types import Command
        for mode in ("stop", "error"):
            ready = asyncio.Event()

            class InterruptedPlan:
                async def astream_events(self, data, *, config=None, version):
                    yield {"event": "on_tool_start", "name": "write_todos", "run_id": "plan", "parent_ids": ["root"], "data": {"input": {}}}
                    yield {"event": "on_tool_end", "run_id": "plan", "parent_ids": ["root"], "data": {"output": Command(update={"todos": [{"content": "还没完成", "status": "in_progress"}]})}}
                    ready.set()
                    if mode == "error":
                        raise RuntimeError("离线故障")
                    await asyncio.Event().wait()

            app = MieApp(InterruptedPlan())
            async with app.run_test() as pilot:
                await self.send(app, pilot, "研究")
                await asyncio.wait_for(ready.wait(), 2)
                if mode == "stop":
                    await pilot.press("escape")
                await self.settle(app)
                text = str(app.query_one(".task-plan", Static).render())
                self.assertIn("0/1", text)
                self.assertIn("进行中", text)
                self.assertIn("本轮未完成", text)
                self.assertEqual(len(app.conversation), 2 if mode == "error" else 0)

    async def test_idle_animation_keeps_layout_and_stops_on_submit(self):
        app = MieApp(DemoAgent())
        async with app.run_test(size=(48, 24)) as pilot:
            welcome = app.query_one(Welcome)
            await pilot.pause()
            initial_frame = welcome.frame
            initial_region = app.query_one(Composer).region
            await pilot.pause(1.35)
            self.assertNotEqual(welcome.frame, initial_frame)
            self.assertEqual(app.query_one(Composer).region, initial_region)
            # 四种表情/呼吸组合保持相同字格尺寸。
            shapes = {tuple(len(line) for line in frame.plain.splitlines())
                      for frame in welcome.frames.values()}
            self.assertEqual(len(shapes), 1)
            await self.send(app, pilot, "你好")
            await self.settle(app)
            self.assertIsNone(welcome.idle_timer)
            self.assertFalse(app.query(Welcome))

    async def test_welcome_compact_layout_then_full_chat(self):
        app = MieApp(DemoAgent())
        async with app.run_test(size=(100, 45)) as pilot:
            await pilot.pause()
            welcome = app.query_one(".welcome")
            bottom = app.query_one("#bottom")
            self.assertLessEqual(bottom.region.y - welcome.region.bottom, 2)
            await pilot.resize_terminal(48, 24)
            await pilot.pause()
            self.assertLessEqual(app.query_one(Composer).region.bottom, 24)
            await pilot.resize_terminal(40, 18)
            await pilot.pause()
            self.assertLessEqual(app.query_one("#bottom").region.bottom, 18)
            await pilot.resize_terminal(48, 24)
            await self.send(app, pilot, "你好")
            await self.settle(app)
            await pilot.pause()
            self.assertFalse(app.query(".welcome"))
            self.assertEqual(app.query_one("#bottom").region.bottom, 24)

    async def send(self, app, pilot, text):
        composer = app.query_one(Composer)
        composer.focus()
        composer.load_text(text)
        await pilot.press("enter")

    async def settle(self, app):
        for _ in range(200):
            if not app.busy:
                return
            await asyncio.sleep(0.02)
        self.fail("本轮未结束")

    async def test_chat_tools_history_and_narrow_screen(self):
        agent = DemoAgent()
        app = MieApp(agent, model_name="离线测试")
        async with app.run_test(size=(88, 30)) as pilot:
            await self.send(app, pilot, "查资料")
            await self.settle(app)
            cards = app.query(Collapsible)
            self.assertEqual(len(cards), 1)
            self.assertTrue(cards.first().collapsed)
            self.assertIn("1 条", cards.first().title)
            await pilot.click("CollapsibleTitle")
            self.assertFalse(cards.first().collapsed)
            self.assertTrue(any("中文回答" in widget.source for widget in app.query(Markdown)))
            await self.send(app, pilot, "接着讲")
            await self.settle(app)
            self.assertEqual(len(agent.inputs[1]["messages"]), 3)
            await pilot.resize_terminal(48, 22)
            await pilot.pause()
            composer = app.query_one(Composer)
            self.assertGreater(composer.region.width, 20)
            self.assertLessEqual(composer.region.bottom, 22)

    async def test_real_deep_agent_events_and_tool_execution(self):
        usage = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{"id": "time1", "name": "get_current_time", "args": {}}], usage_metadata=usage),
            AIMessage(content="时间工具已执行。", usage_metadata=usage),
        ])
        graph = create_deep_agent(model=model, tools=[get_current_time])
        app = MieApp(graph)
        async with app.run_test() as pilot:
            await self.send(app, pilot, "今天星期几？")
            await self.settle(app)
            self.assertEqual(app.phase, "完成")
            self.assertEqual(len(app.conversation), 4)
            self.assertIn("北京时间", app.query_one(Collapsible).title)
            self.assertIn("星期", app.query_one(Collapsible).title)
            self.assertFalse(app.query(".task-plan"))
            self.assertEqual(app.stats.output_tokens, 20)
            self.assertNotIn("— token/s", app.stats.summary())

    async def test_openai_compatible_stream_and_usage_without_network(self):
        def response(request):
            body = json.loads(request.content)
            self.assertTrue(body["stream"])
            self.assertTrue(body["stream_options"]["include_usage"])
            chunks = []
            for delta, finish, usage in (
                ({"role": "assistant", "content": "你好"}, None, None),
                ({"content": "，小咩！"}, None, None),
                ({}, "stop", {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}),
            ):
                chunks.append("data: " + json.dumps({"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "offline", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage}) + "\n\n")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="".join(chunks) + "data: [DONE]\n\n")

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            model = ChatOpenAI(model="offline", api_key="offline-test", base_url="https://example.invalid/v1", http_async_client=client, stream_usage=True)
            app = MieApp(create_deep_agent(model=model))
            async with app.run_test() as pilot:
                await self.send(app, pilot, "你好")
                await self.settle(app)
                self.assertEqual(app.phase, "完成")
                self.assertEqual(app.conversation[-1].content, "你好，小咩！")
                self.assertEqual(app.stats.output_tokens, 9)
                self.assertEqual(len(app.query(Markdown)), 1)

    async def test_scrolling_does_not_jump_and_stream_can_be_stopped(self):
        first = asyncio.Event()
        more = asyncio.Event()
        long_text = "\n\n".join(f"第 {n} 段：中文研究资料。" for n in range(50))

        class StreamingAgent:
            async def astream_events(self, data, *, config=None, version):
                yield {"event": "on_chat_model_start", "run_id": "model", "parent_ids": ["root"], "data": {}}
                yield {"event": "on_chat_model_stream", "run_id": "model", "parent_ids": ["root"], "data": {"chunk": AIMessageChunk(content=long_text)}}
                first.set()
                await more.wait()
                yield {"event": "on_chat_model_stream", "run_id": "model", "parent_ids": ["root"], "data": {"chunk": AIMessageChunk(content="\n\n新增内容。")}}
                await asyncio.Event().wait()

        app = MieApp(StreamingAgent())
        async with app.run_test(size=(60, 26)) as pilot:
            await self.send(app, pilot, "长回答")
            await asyncio.wait_for(first.wait(), 2)
            await pilot.pause()
            await pilot.press("pageup")
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            position = chat.scroll_y
            more.set()
            await pilot.pause()
            self.assertLess(chat.scroll_y, chat.max_scroll_y)
            self.assertAlmostEqual(chat.scroll_y, position, delta=1)
            await pilot.press("escape")
            await self.settle(app)
            self.assertEqual(app.phase, "已停止")
            self.assertEqual(app.conversation, [])
            self.assertIn("新增内容", app.query_one(Markdown).source)
            self.assertIn("未提供", app.stats.summary(final=True))

    async def test_cancel_keeps_completed_history_and_allows_next_turn(self):
        agent = DemoAgent("waiting")
        app = MieApp(agent)
        app.conversation = [HumanMessage(content="之前的问题"), AIMessage(content="之前的回答")]
        async with app.run_test() as pilot:
            await self.send(app, pilot, "等待搜索")
            await asyncio.wait_for(agent.waiting.wait(), 2)
            await pilot.press("escape")
            await self.settle(app)
            self.assertEqual(len(app.conversation), 2)
            self.assertIn("停止", app.query_one(Collapsible).title)
            agent.mode = "success"
            await self.send(app, pilot, "继续")
            await self.settle(app)
            self.assertEqual(len(agent.inputs[-1]["messages"]), 3)

    async def test_error_and_multiline_input(self):
        agent = DemoAgent("error")
        app = MieApp(agent)
        async with app.run_test() as pilot:
            composer = app.query_one(Composer)
            await pilot.press("a", "ctrl+j", "b")
            self.assertEqual(composer.text, "a\nb")
            await pilot.press("enter")
            await self.settle(app)
            self.assertEqual(app.conversation[0], {"role": "user", "content": "a\nb"})
            self.assertIn("本轮执行失败", app.conversation[1]["content"])
            self.assertIn("失败", app.query_one(Collapsible).title)
            agent.mode = "success"
            await self.send(app, pilot, "重新提问")
            await self.settle(app)
            self.assertEqual(len(agent.inputs[-1]["messages"]), 3)
            self.assertEqual(agent.inputs[-1]["messages"][0]["content"], "a\nb")

    async def test_failed_research_survives_connection_check_and_retry(self):
        agent = DemoAgent("error")
        app = MieApp(agent)
        async with app.run_test() as pilot:
            await self.send(app, pilot, "比较两个框架，只查官方资料")
            await self.settle(app)
            self.assertEqual(len(app.conversation), 2)
            agent.mode = "success"
            await self.send(app, pilot, "只回复连接正常")
            await self.settle(app)
            await self.send(app, pilot, "重新执行之前的研究任务")
            await self.settle(app)
            history = agent.inputs[-1]["messages"]
            self.assertEqual(history[0]["content"], "比较两个框架，只查官方资料")
            self.assertIn("本轮执行失败", history[1]["content"])
            self.assertFalse(any(isinstance(m, ToolMessage) for m in history))
            self.assertFalse(any(getattr(m, "tool_calls", None) for m in history))
            await self.send(app, pilot, "/clear")
            self.assertEqual(app.conversation, [])

    def test_connection_diagnostic_is_useful_without_exposing_secrets(self):
        import socket
        from app.tui import error_diagnostic
        inner = socket.gaierror("secret-password URL with credentials")
        outer = RuntimeError("Bearer private-key")
        outer.__cause__ = inner
        result = error_diagnostic(outer)
        self.assertIn("RuntimeError → gaierror", result)
        self.assertIn("域名解析失败", result)
        self.assertNotIn("private-key", result)
        self.assertNotIn("secret-password", result)
        outer.__cause__ = httpx.ReadTimeout("private-url")
        self.assertIn("请求超时", error_diagnostic(outer))
        outer.__cause__ = httpx.RemoteProtocolError("private-body")
        self.assertIn("连接在收发数据时中断", error_diagnostic(outer))

    async def test_local_commands_do_not_call_model(self):
        agent = DemoAgent()
        app = MieApp(agent)
        async with app.run_test() as pilot:
            for command in ("", "/help", "/clear"):
                await self.send(app, pilot, command)
            self.assertEqual(agent.inputs, [])
            self.assertEqual(app.conversation, [])

    async def test_fast_typing_and_paste_keep_all_characters(self):
        agent = DemoAgent()
        app = MieApp(agent)
        async with app.run_test() as pilot:
            composer = app.query_one(Composer)
            for char in "/help":
                app.post_message(events.Key(char, char))
            app.post_message(events.Key("enter", "\r"))
            await pilot.pause()
            self.assertEqual(agent.inputs, [])
            self.assertTrue(any("token/s =" in str(w.render()) for w in app.query(Static)))
            app.post_message(events.Paste("中文第一行\n第二行"))
            await pilot.pause()
            self.assertEqual(composer.text, "中文第一行\n第二行")
            self.assertEqual(agent.inputs, [])

    def test_token_rate_uses_reported_usage_and_excludes_tool_wait(self):
        stats = UsageStats()
        stats.start("a", 0)
        stats.finish("a", AIMessage(content="很长的文本" * 100, usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}), 2)
        # 两次模型调用之间的 20 秒搜索时间不进入分母。
        stats.start("b", 22)
        stats.finish("b", AIMessage(content="短回复", usage_metadata={"input_tokens": 30, "output_tokens": 20, "total_tokens": 50}), 26)
        self.assertEqual(stats.output_tokens, 30)
        self.assertIn("5.0 token/s", stats.summary())
        stats.start("missing", 27)
        stats.finish("missing", AIMessage(content="无用量数据"), 28)
        self.assertIn("部分", stats.summary())
        self.assertNotIn("5.0 token/s", stats.summary())


if __name__ == "__main__":
    unittest.main()
