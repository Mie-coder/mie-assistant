"""可选游戏模块的加载边界；子进程避免其他游戏测试提前 import 掩盖问题。"""
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FeatureLoadingTests(unittest.TestCase):
    def isolated(self, source, enabled=None):
        env = dict(os.environ, LANGSMITH_TRACING="false", PYTHONDONTWRITEBYTECODE="1")
        if enabled is None:
            env.pop("MIE_ENABLE_2048", None)
        else:
            env["MIE_ENABLE_2048"] = enabled
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_startup_does_not_import_game_runtime_in_either_mode(self):
        for enabled in (None, "0", "1"):
            with self.subTest(enabled=enabled):
                self.isolated('''
                    import importlib.abc
                    import os
                    import sys
                    from unittest.mock import patch
                    class BlockRuntime(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path, target=None):
                            if fullname in {"app.game2048", "app.game2048_solver",
                                            "app.features.game2048.runtime", "app.features.game2048.solver"}:
                                raise AssertionError("Startup loaded game implementation: " + fullname)
                    sys.meta_path.insert(0, BlockRuntime())
                    from app.agent import build_agent
                    with patch.dict(os.environ, {"AGENTSEEK_MODEL": "offline", "OPENAI_API_KEY": "offline",
                                                "OPENAI_API_BASE": "https://example.invalid"}), \\
                         patch("app.agent.ChatOpenAI"), patch("app.agent.create_deep_agent") as create:
                        build_agent()
                    config = create.call_args.kwargs
                    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in config["tools"]}
                    enabled = os.environ.get("MIE_ENABLE_2048", "1") == "1"
                    assert ("play_2048" in names) == enabled
                    assert ("play_2048" in config["interrupt_on"]) == enabled
                    assert "2048" not in config["system_prompt"]
                    assert "棋盘" not in config["system_prompt"]
                    result = config["backend"].read("/skills/game2048/SKILL.md")
                    assert result.error is None
                    skill = result.file_data["content"]
                    assert "name: game2048" in skill
                    assert "continuation_id" in skill
                    if enabled:
                        game_tool = next(t for t in config["tools"] if getattr(t, "name", "") == "play_2048")
                        assert "/skills/game2048/SKILL.md" in game_tool.description
                    assert {"get_current_time", "internet_search", "jev_choose"} <= names
                    assert {"execute", "write_file", "edit_file", "jev_choose"} <= config["interrupt_on"].keys()
                    if not enabled:
                        assert "app.features.game2048" not in sys.modules
                ''', enabled)

    def test_approval_is_required_before_lazy_loading(self):
        for verdict in ("reject", "approve"):
            with self.subTest(verdict=verdict):
                self.isolated('''
                    import asyncio
                    import sys
                    from unittest.mock import AsyncMock, patch
                    from langchain_core.messages import AIMessage, ToolMessage
                    from langgraph.types import Command
                    from app.agent import build_agent
                    from tests.test_tui import ScriptedModel
                    import app.features.game2048 as feature
                    assert "app.features.game2048.runtime" not in sys.modules
                    model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[{
                        "name": "play_2048", "id": "g1", "args": {"max_steps": 1, "max_seconds": 20},
                    }]), AIMessage(content="结束")])
                    with patch.dict("os.environ", {"AGENTSEEK_MODEL": "offline", "OPENAI_API_KEY": "offline",
                                                   "OPENAI_API_BASE": "https://example.invalid"}), \\
                         patch("app.agent.ChatOpenAI", return_value=model):
                        graph = build_agent()
                    async def check():
                        config = {"configurable": {"thread_id": "lazy-approval"}}
                        fake = AsyncMock(return_value={"ok": True, "reason": "step_limit", "steps": 1})
                        with patch.object(feature, "_load_runner", return_value=fake) as load, \\
                             patch("app.jev.jev_session"):
                            pending = await graph.ainvoke({"messages": [{"role": "user", "content": "玩一步 2048"}]}, config)
                            load.assert_not_called()
                            pause = pending["__interrupt__"][0]
                            await graph.ainvoke(Command(resume={pause.id: {"decisions": [{"type": VERDICT}]}}), config)
                        assert load.call_count == (1 if VERDICT == "approve" else 0)
                        assert fake.await_count == (1 if VERDICT == "approve" else 0)
                    asyncio.run(check())
                '''.replace("VERDICT", repr(verdict)))

    def test_disabled_direct_calls_do_not_load_runtime(self):
        self.isolated('''
            import asyncio
            from unittest.mock import patch
            from app.features import game2048 as feature
            with patch.object(feature, "_load_runner", side_effect=AssertionError("Must not load")):
                assert feature.play_2048.invoke({})["reason"] == "feature_disabled"
                assert asyncio.run(feature.play_2048.ainvoke({}))["reason"] == "feature_disabled"
        ''', "0")

    def test_missing_optional_resources_fail_at_invocation_only(self):
        self.isolated('''
            import asyncio
            from unittest.mock import patch
            from app.features import game2048 as feature
            for error in (ImportError("private-path"), FileNotFoundError("private-path")):
                with patch.object(feature, "_load_runner", side_effect=error), patch("app.jev.jev_session") as session:
                    result = asyncio.run(feature.play_2048.ainvoke({}))
                    assert result["ok"] is False
                    assert result["reason"] == "feature_unavailable"
                    assert "private-path" not in str(result)
                    session.assert_not_called()
        ''')

    def test_sync_and_async_calls_preserve_arguments_and_workspace_paths(self):
        self.isolated('''
            import asyncio
            from pathlib import Path
            from unittest.mock import AsyncMock, patch
            from app.features import game2048 as feature
            from app.features.game2048 import runtime, solver
            expected = Path.cwd() / "workspace"
            assert runtime.REPORT_DIR == expected / "reports" / "2048"
            assert runtime.LESSONS_PATH == expected / "memories" / "2048-lessons.md"
            assert solver.CACHE == expected / ".cache" / "2048"
            assert (solver.SOURCE / "LICENSE").is_file()
            assert "play2048.co" in runtime.PAGE_SCRIPT
            args = dict(max_steps=2, max_seconds=20, search_assist=True, new_game=True,
                        keep_page_open=True, continuation_id="same-session")
            fake = AsyncMock(return_value={"ok": True, "steps": 2})
            with patch.object(feature, "_load_runner", return_value=fake), patch("app.jev.jev_session"):
                assert feature.play_2048.invoke(args)["steps"] == 2
                assert asyncio.run(feature.play_2048.ainvoke(args))["steps"] == 2
            assert fake.await_count == 2
            assert fake.await_args.args == (2, 20, True, True, True, "same-session")
        ''')
