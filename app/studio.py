"""Studio / Agent Server 入口，复用小咩的工具、提示词和审批规则。"""

import os

from app.agent import build_agent

# .env 由 langgraph.json 加载，不经过终端的 main()。
os.environ.setdefault("LANGSMITH_PROJECT", "mie-assistant")
graph = build_agent(use_local_checkpointer=False)
