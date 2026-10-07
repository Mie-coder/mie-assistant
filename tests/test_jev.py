"""Jev 离线契约测试；不发送网络请求。"""
import io
import json
import unittest
import asyncio
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import httpx

from app.jev import jev_choose, jev_session


class JevTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.options = {"left": "Move left", "stop": "Insufficient evidence"}

    def invoke(self):
        return jev_choose("Observed board", "Which action?", self.options)

    def test_request_and_structured_response(self):
        response = {"model": "jev-latest", "answers": {"decision": {
            "type": "choice", "choice": "left", "probabilities": {"left": .8, "stop": .2}, "confidence": .5}},
            "usage": {"input_tokens": 10, "output_tokens": 3}}
        with patch("app.jev.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            result = self.invoke()
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(json.loads(request.data)["questions"]["decision"]["criteria"], self.options)
        self.assertTrue(result["ok"])
        self.assertEqual(result["choice"], "left")
        self.assertEqual(result["usage"], response["usage"])

    def test_missing_key_does_not_send(self):
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": ""}), patch("app.jev.urlopen") as send:
            self.assertEqual(self.invoke()["error"], "not_configured")
            send.assert_not_called()

    def test_invalid_inputs_do_not_send(self):
        with patch("app.jev.urlopen") as send:
            self.assertFalse(jev_choose("", "Question", self.options)["ok"])
            self.assertFalse(jev_choose("State", "Question", {"a": "only"})["ok"])
            send.assert_not_called()

    def test_failures_are_bounded_and_do_not_echo_secrets(self):
        for error in (HTTPError("url", 401, "test-secret", {}, None), URLError("test-secret"), TimeoutError()):
            with self.subTest(error=type(error)), patch("app.jev.urlopen", side_effect=error) as send:
                result = self.invoke()
                self.assertFalse(result["ok"])
                self.assertNotIn("test-secret", json.dumps(result))
                self.assertEqual(send.call_count, 1)

    def test_invalid_response_never_returns_a_decision(self):
        for raw in (b"not json", b'{}', b'{"answers":{"decision":{"choice":"delete"}}}'):
            with patch("app.jev.urlopen", return_value=io.BytesIO(raw)):
                self.assertEqual(self.invoke()["error"], "invalid_response")

    def test_game_session_reuses_client_in_worker_thread_and_closes_it(self):
        requests = []
        response = {"answers": {"decision": {"type": "choice", "choice": "left",
                    "probabilities": {"left": .8, "stop": .2}, "confidence": .5}}}
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json=response)
        client = httpx.Client(transport=httpx.MockTransport(handle))
        async def choose_twice():
            with jev_session():
                for _ in range(2):
                    self.assertTrue((await asyncio.to_thread(self.invoke))["ok"])
        with patch("app.jev.httpx.Client", return_value=client) as factory:
            asyncio.run(choose_twice())
        factory.assert_called_once()
        self.assertTrue(client.is_closed)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test-secret")
        with patch("app.jev.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            self.assertTrue(self.invoke()["ok"])
            send.assert_called_once()  # The scoped client does not leak to other tools.

    def test_pooled_failures_do_not_retry_or_echo_response(self):
        for code in (401, 429, 500):
            calls = []
            def handle(request):
                calls.append(request)
                return httpx.Response(code, text="test-secret")
            client = httpx.Client(transport=httpx.MockTransport(handle))
            with patch("app.jev.httpx.Client", return_value=client), jev_session():
                result = self.invoke()
            self.assertEqual(result["error"], "http_error")
            self.assertEqual(result["status"], code)
            self.assertNotIn("test-secret", json.dumps(result))
            self.assertEqual(len(calls), 1)

    def test_actual_agent_requires_approval_and_reject_does_not_send(self):
        from app.agent import build_agent
        from tests.test_tui import ScriptedModel
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.types import Command
        for decision in ("approve", "reject"):
            model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[{
                "name": "jev_choose", "id": "j1", "args": {
                    "state": "Observed state", "question": "Which action?", "choices": self.options},
            }]), AIMessage(content="结束")])
            with patch.dict("os.environ", {"AGENTSEEK_MODEL": "offline", "OPENAI_API_KEY": "offline",
                                           "OPENAI_API_BASE": "https://example.invalid"}), patch("app.agent.ChatOpenAI", return_value=model):
                graph = build_agent()
            config = {"configurable": {"thread_id": decision}}
            response = {"answers": {"decision": {"type": "choice", "choice": "left",
                        "probabilities": {"left": .8, "stop": .2}, "confidence": .5}}}
            with patch("app.jev.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
                state = graph.invoke({"messages": [{"role": "user", "content": "用 Jev 判断"}]}, config=config)
                send.assert_not_called()
                pause = state["__interrupt__"][0]
                self.assertEqual(pause.value["action_requests"][0]["name"], "jev_choose")
                result = graph.invoke(Command(resume={pause.id: {"decisions": [{"type": decision}]}}), config=config)
                self.assertEqual(send.call_count, int(decision == "approve"))
                if decision == "approve":
                    output = next(m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "jev_choose")
                    self.assertEqual(json.loads(output.content)["choice"], "left")


if __name__ == "__main__":
    unittest.main()
