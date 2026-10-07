"""由 Agent Server 调度的研究员；只提供搜索，不授予主助手的 Shell/审批工具。"""

from langchain.agents import create_agent

from app.agent import build_model
from app.research import RESEARCH_PROMPT, InterruptedResearchHistory
from app.tools import internet_search


graph = create_agent(
    model=build_model(), tools=[internet_search], system_prompt=RESEARCH_PROMPT,
    middleware=[InterruptedResearchHistory()],
)
