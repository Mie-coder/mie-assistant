"""中断更新不得将缺少工具结果的历史发给模型，也不得重放未完成操作。"""

import unittest

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.research import InterruptedResearchHistory
from tests.test_tui import ScriptedModel


class ResearchHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_keeps_complete_evidence_and_new_request(self):
        executed = []

        def search(query: str) -> str:
            """测试用搜索记录。"""
            executed.append(query)
            return "搜索结果"

        good = AIMessage(content="", tool_calls=[{"id": "good", "name": "search", "args": {"query": "已完成"}}])
        pending = AIMessage(content="", tool_calls=[
            {"id": "p1", "name": "search", "args": {"query": "部分完成"}},
            {"id": "p2", "name": "search", "args": {"query": "未完成"}},
        ])
        history = [HumanMessage(content="原研究主题"), good,
                   ToolMessage("可信来源 URL", tool_call_id="good"), pending,
                   ToolMessage("只返回了一项", tool_call_id="p1"),
                   HumanMessage(content="追加：只看官方文档")]
        graph = create_agent(model=ScriptedModel(responses=[AIMessage(content="按新要求继续")]),
                             tools=[search], middleware=[InterruptedResearchHistory()])
        result = await graph.ainvoke({"messages": history})
        messages = result["messages"]
        self.assertEqual([m.content for m in messages if isinstance(m, HumanMessage)],
                         ["原研究主题", "追加：只看官方文档"])
        self.assertEqual([m.tool_call_id for m in messages if isinstance(m, ToolMessage)], ["good"])
        self.assertEqual([c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls], ["good"])
        self.assertEqual(executed, [])
