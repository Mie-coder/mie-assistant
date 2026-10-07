"""服务集成测试专用图：等待后回显整段历史和固定来源，不调用模型。"""

import asyncio

from langgraph.graph import END, START, MessagesState, StateGraph


async def research(state):
    await asyncio.sleep(8)
    instructions = "\n".join(str(msg.content) for msg in state["messages"] if msg.type == "human")
    return {"messages": [{"role": "ai", "content": instructions + "\n测试来源：https://docs.langchain.com/"}]}


builder = StateGraph(MessagesState)
builder.add_node("research", research)
builder.add_edge(START, "research")
builder.add_edge("research", END)
graph = builder.compile()
