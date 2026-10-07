"""TypeSafe Jev Choice 接口；只提供建议，不执行浏览器动作。"""
import json
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, getproxies, proxy_bypass, urlopen

import httpx

_session_client = ContextVar("jev_session_client", default=None)


@contextmanager
def jev_session():
    """一次游戏复用连接；to_thread 继承上下文，结束后不影响独立工具调用。"""
    proxy = getproxies().get("https") if not proxy_bypass("api.typesafe.ai") else None
    with httpx.Client(timeout=30, proxy=proxy, limits=httpx.Limits(
            max_connections=1, max_keepalive_connections=1, keepalive_expiry=30)) as client:
        token = _session_client.set(client)
        try:
            yield
        finally:
            _session_client.reset(token)


def jev_choose(state: str, question: str, choices: dict[str, str]) -> dict:
    """将观察到的文字状态交给 Jev，从给定选项中选择一个，返回概率和置信度。

    会把传入内容发送至 TypeSafe API，需要 TYPESAFE_API_KEY。
    state 是最新网页文字或已确认的 JSON 状态字符串，不接受图片或截图路径。
    question 只问一个具体判断；choices 为选项标识到说明的映射，至少两个。
    建议用英文问题与选项说明，并加入 stop/unknown 供证据不足时选择。
    返回 ok=false 时停止依赖此决策的操作，不编造选择，不自动重试。
    本工具不操作网页；需要操作时由 BrowserSkill 执行，并重新观察验证。
    """
    if (not state.strip() or not question.strip() or len(choices) < 2
            or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) for k, v in choices.items())):
        return {"ok": False, "error": "invalid_input", "message": "需要非空状态、问题和至少两个文字选项。"}
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        return {"ok": False, "error": "not_configured", "message": "请在小咩 .env 中填写 TYPESAFE_API_KEY 并重启。"}
    body = json.dumps({
        "model": "jev-latest", "state": state,
        "questions": {"decision": {"type": "choice", "instructions": question, "criteria": choices}},
    }, ensure_ascii=False).encode("utf-8")
    if len(body) > 100_000:
        return {"ok": False, "error": "input_too_large", "message": "请提取与当前判断有关的状态，限制请求在 100 KB 内。"}
    request = Request("https://api.typesafe.ai/v1/systemone", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    }, method="POST")
    try:
        # 单次调用，不因重试产生额外用量。错误不回显响应正文或凭证。
        client = _session_client.get()
        if client is None:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read(1_000_001))
        else:
            with client.stream("POST", request.full_url, content=body,
                               headers=dict(request.header_items())) as response:
                response.raise_for_status()
                raw = bytearray()
                for chunk in response.iter_bytes(65_536):
                    raw.extend(chunk)
                    if len(raw) > 1_000_000:
                        raise ValueError("Response too large")
                result = json.loads(raw)
    except (HTTPError, httpx.HTTPStatusError) as exc:
        status = exc.code if isinstance(exc, HTTPError) else exc.response.status_code
        return {"ok": False, "error": "http_error", "status": status,
                "message": "Jev 请求失败；检查密钥、额度或服务状态，不要自动重试。"}
    except (URLError, TimeoutError, OSError, httpx.RequestError):
        return {"ok": False, "error": "connection_error", "message": "Jev 连接失败或超时，未获得决策。"}
    except (ValueError, UnicodeError):
        return {"ok": False, "error": "invalid_response", "message": "Jev 响应不是有效 JSON，未获得决策。"}
    try:
        answer = result["answers"]["decision"]
        probabilities, confidence = answer["probabilities"], answer["confidence"]
        def probability(value):
            return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
        if (answer["type"] != "choice" or answer["choice"] not in choices
                or set(probabilities) != set(choices) or not probability(confidence)
                or not all(probability(p) for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=.01)):
            raise ValueError("Invalid choice")
        return {"ok": True, "choice": answer["choice"], "probabilities": probabilities,
                "confidence": confidence, "model": result.get("model"), "usage": result.get("usage", {})}
    except (KeyError, TypeError, ValueError, AttributeError):
        return {"ok": False, "error": "invalid_response", "message": "Jev 决策字段不完整或超出选项范围，未获得有效决策。"}
