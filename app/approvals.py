"""本轮审批策略与 LangGraph 中断协议；不依赖 Textual，不执行工具。"""

from dataclasses import dataclass
from typing import Awaitable, Callable

from langgraph.types import Command


@dataclass(frozen=True)
class ApprovalRequest:
    name: str
    arguments: dict
    allowed: tuple[str, ...]


class TurnApprovals:
    """只保存当前轮的选择；调用方必须在完成、失败或取消时 reset。"""

    def __init__(self):
        self.automatic = False

    def toggle(self):
        self.automatic = not self.automatic

    def reset(self):
        self.automatic = False

    def can_auto_approve(self, request: ApprovalRequest) -> bool:
        return self.automatic and "approve" in request.allowed

    async def resume(
        self,
        interrupts,
        review: Callable[[ApprovalRequest], Awaitable[str]],
    ) -> Command:
        replies = {}
        for pause in interrupts:
            payload = pause.value
            if not isinstance(payload, dict):
                raise ValueError("无法识别的审批请求，未自动放行。")
            actions = payload.get("action_requests", [])
            configs = payload.get("review_configs", [])
            if not actions or len(actions) != len(configs):
                raise ValueError("审批请求与规则不匹配，未自动放行。")
            decisions = []
            for action, config in zip(actions, configs):
                if action["name"] != config["action_name"]:
                    raise ValueError("审批工具与规则不匹配，未自动放行。")
                request = ApprovalRequest(
                    action["name"], action["args"], tuple(config["allowed_decisions"]),
                )
                if not {"approve", "reject"}.intersection(request.allowed):
                    raise ValueError("此请求不支持批准或拒绝，已停止。")
                choice = await review(request)
                if choice not in ("approve", "reject") or choice not in request.allowed:
                    raise ValueError("不支持的审批决定，未执行。")
                decision = {"type": choice}
                if choice == "reject":
                    decision["message"] = "用户拒绝了此操作，请勿换一种方式重试。"
                decisions.append(decision)
            replies[pause.id] = {"decisions": decisions}
        return Command(resume=replies)
