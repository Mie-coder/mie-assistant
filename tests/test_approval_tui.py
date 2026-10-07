"""审批与模式切换：真实 LangGraph 中断，本地模型和可计数的无副作用工具。"""
import asyncio
import unittest

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from textual.widgets import Button

from tests.test_tui import ScriptedModel, DemoAgent
from app.tui import MieApp, Composer
from app.approvals import TurnApprovals


def call(command, call_id):
    return AIMessage(content="", tool_calls=[
        {"name": "execute", "args": {"command": command}, "id": call_id}
    ])


def graph_for(responses, executed):
    def execute(command: str) -> str:
        """记录测试命令，不访问 Shell。"""
        executed.append(command)
        return "执行结果：" + command

    return create_deep_agent(
        model=ScriptedModel(responses=responses), tools=[execute],
        checkpointer=InMemorySaver(),
        interrupt_on={"execute": {"allowed_decisions": ["approve", "reject"]}},
    )


class ApprovalTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_back_to_manual_before_next_tool(self):
        executed = []
        started, release = asyncio.Event(), asyncio.Event()

        async def execute(command: str) -> str:
            """只记录，不执行 Shell。"""
            executed.append(command)
            if command == "first":
                started.set()
                await release.wait()
            return command

        graph = create_deep_agent(
            model=ScriptedModel(responses=[call("first", "c1"), call("second", "c2"), AIMessage(content="结束")]),
            tools=[execute], checkpointer=InMemorySaver(),
            interrupt_on={"execute": {"allowed_decisions": ["approve", "reject"]}},
        )
        app = MieApp(graph)
        async with app.run_test() as pilot:
            await pilot.press("tab")
            await self.send(app, pilot)
            await asyncio.wait_for(started.wait(), 2)
            await pilot.press("tab")
            release.set()
            await self.wait_for(lambda: app.approval_pending)
            self.assertEqual(executed, ["first"])
            await pilot.press("ctrl+n")
            await self.wait_for(lambda: not app.busy)
            self.assertEqual(executed, ["first"])

    async def test_protocol_preserves_multiple_interrupt_ids_and_rejects_unknown(self):
        policy = TurnApprovals()
        policy.toggle()
        async def review(request):
            return "approve" if policy.can_auto_approve(request) else "reject"
        def pause(key, names):
            return Interrupt(id=key, value={
                "action_requests": [{"name": name, "args": {}} for name in names],
                "review_configs": [{"action_name": name, "allowed_decisions": ["approve", "reject"]} for name in names],
            })
        result = await policy.resume([pause("a", ["execute", "write_file"]), pause("b", ["edit_file"])], review)
        self.assertEqual(result.resume, {
            "a": {"decisions": [{"type": "approve"}, {"type": "approve"}]},
            "b": {"decisions": [{"type": "approve"}]},
        })
        with self.assertRaises(ValueError):
            await policy.resume([Interrupt(id="unknown", value="unexpected")], review)

    async def wait_for(self, condition):
        for _ in range(200):
            if condition():
                return
            await asyncio.sleep(.02)
        self.fail("界面未进入预期状态")

    async def send(self, app, pilot, text="测试审批"):
        app.query_one(Composer).focus()
        app.query_one(Composer).load_text(text)
        await pilot.press("enter")

    async def test_manual_approve_then_reject_never_executes_rejected_action(self):
        executed = []
        graph = graph_for([call("first", "c1"), call("second", "c2"), AIMessage(content="结束")], executed)
        app = MieApp(graph)
        async with app.run_test(size=(48, 24)) as pilot:
            await self.send(app, pilot)
            await self.wait_for(lambda: app.approval_pending)
            self.assertEqual(executed, [])
            await pilot.press("ctrl+y")
            await self.wait_for(lambda: executed == ["first"] and app.approval_pending)
            await pilot.press("ctrl+n")
            await self.wait_for(lambda: not app.busy)
            self.assertEqual(executed, ["first"])
            self.assertEqual(app.phase, "完成")
            self.assertFalse(app.approvals.automatic)
            self.assertTrue(any(isinstance(m, ToolMessage) and "拒绝" in m.content for m in app.conversation))
            self.assertLessEqual(app.query_one(Composer).region.bottom, 24)
            self.assertFalse(app.query(".approval-card Button"))
            # 取消/完成后的检查点不留在内存中。
            self.assertEqual(graph.checkpointer.storage, {})

    async def test_tab_auto_is_one_turn_only_and_clear_removes_history(self):
        executed = []
        graph = graph_for([call("one", "c1"), call("two", "c2"), AIMessage(content="第一轮"),
                           call("three", "c3"), AIMessage(content="第二轮"), AIMessage(content="新会话")], executed)
        app = MieApp(graph)
        async with app.run_test() as pilot:
            app.query_one(Composer).load_text("保留输入")
            await pilot.press("tab")
            self.assertTrue(app.approvals.automatic)
            self.assertEqual(app.query_one(Composer).text, "保留输入")
            await self.send(app, pilot)
            await self.wait_for(lambda: not app.busy)
            self.assertEqual(executed, ["one", "two"])
            self.assertFalse(app.approvals.automatic)
            first_history = list(app.conversation)
            await self.send(app, pilot, "第二轮")
            await self.wait_for(lambda: app.approval_pending)
            self.assertEqual(executed, ["one", "two"])
            await pilot.press("ctrl+n")
            await self.wait_for(lambda: not app.busy)
            self.assertEqual(app.conversation[:len(first_history)], first_history)
            self.assertEqual(sum(isinstance(m, HumanMessage) for m in app.conversation), 2)
            await self.send(app, pilot, "/clear")
            self.assertEqual(app.conversation, [])
            await self.send(app, pilot, "新的问题")
            await self.wait_for(lambda: not app.busy)
            self.assertEqual(len(app.conversation), 2)

    async def test_switch_to_auto_while_pending_and_cancel_does_not_resume_old_call(self):
        executed = []
        graph = graph_for([call("cancelled", "c0"), call("allowed", "c1"), AIMessage(content="结束")], executed)
        app = MieApp(graph)
        async with app.run_test() as pilot:
            await self.send(app, pilot)
            await self.wait_for(lambda: app.approval_pending)
            await pilot.press("escape")
            await self.wait_for(lambda: not app.busy)
            self.assertEqual(executed, [])
            self.assertEqual(app.conversation, [])
            self.assertFalse(app.approvals.automatic)
            await self.send(app, pilot, "重新开始")
            await self.wait_for(lambda: app.approval_pending)
            await pilot.press("tab")
            await self.wait_for(lambda: not app.busy)
            self.assertEqual(executed, ["allowed"])
            self.assertFalse(app.approvals.automatic)

    async def test_auto_resets_on_failure_and_stop(self):
        for mode in ("error", "waiting"):
            app = MieApp(DemoAgent(mode))
            async with app.run_test() as pilot:
                await pilot.press("tab")
                await self.send(app, pilot)
                if mode == "waiting":
                    await asyncio.wait_for(app.agent.waiting.wait(), 2)
                    await pilot.press("escape")
                await self.wait_for(lambda: not app.busy)
                self.assertFalse(app.approvals.automatic)
                self.assertEqual(len(app.conversation), 2 if mode == "error" else 0)


if __name__ == "__main__":
    unittest.main()
