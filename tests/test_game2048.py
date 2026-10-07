"""2048 插件离线验收：模拟浏览器和 Jev，不使用真实额度。"""
import asyncio
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.features.game2048 import runtime as game


def state(board=None):
    return {"id": "test-game", "state": "playing", "score": 0, "moveCount": 0,
            "board": board or [[2, 2, 0, 0], [0]*4, [0]*4, [0]*4]}


class FakeBrowser:
    def __init__(self):
        self.current = state()
        self.moves = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def read(self):
        return copy.deepcopy(self.current)

    async def move(self, direction, expected):
        if self.current != expected:
            raise game.GameError("board_changed")
        self.moves.append(direction)
        board, gain = game.slide(self.current["board"], direction)
        empty = next((y, x) for y in range(4) for x in range(4) if board[y][x] == 0)
        board[empty[0]][empty[1]] = 2
        self.current.update(board=board, score=self.current["score"] + gain,
                            moveCount=self.current["moveCount"] + 1)
        return copy.deepcopy(self.current)

    async def screenshot(self, path):
        return None

    async def close(self):
        self.closed = True


class BoardTests(unittest.TestCase):
    def test_merge_once_in_each_direction(self):
        board = [[2, 2, 4, 4], [4, 4, 4, 0], [0]*4, [0]*4]
        left, gain = game.slide(board, "left")
        self.assertEqual(left[:2], [[4, 8, 0, 0], [8, 4, 0, 0]])
        self.assertEqual(gain, 20)
        right, _ = game.slide(board, "right")
        self.assertEqual(right[1], [0, 0, 4, 8])
        vertical = [[2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]]
        self.assertEqual(game.slide(vertical, "up")[0], [[4, 0, 0, 0], [4, 0, 0, 0], [0]*4, [0]*4])
        self.assertEqual(game.slide(vertical, "down")[0], [[0]*4, [0]*4, [4, 0, 0, 0], [4, 0, 0, 0]])
        self.assertEqual(board[0], [2, 2, 4, 4])

    def test_reject_incomplete_or_invalid_board(self):
        for board in ([[2]*4]*3, [[2]*3]*4, [[3]*4]*4, [[True]*4]*4, [[-2]*4]*4):
            with self.subTest(board=board), self.assertRaises(game.GameError):
                game.validate_state(state(board))

    def test_verify_requires_real_slide_plus_one_spawn(self):
        before = state()
        after = state([[4, 0, 0, 0], [2, 0, 0, 0], [0]*4, [0]*4])
        after.update(score=4, moveCount=1)
        game.verify_move(before, after, "left")
        for change in ({"moveCount": 0}, {"score": 0}, {"id": "other-game"},
                       {"board": [[4, 4, 4, 0], [0]*4, [0]*4, [0]*4]}):
            with self.subTest(change=change), self.assertRaises(game.GameError):
                game.verify_move(before, {**after, **change}, "left")


class SafetyFilterTests(unittest.TestCase):
    """阶段二：安全方向计算与「只把安全方向给 Jev」的离线验证。"""

    # 第 2 局（game-3faa21bd…）最后一步的真实棋盘：满盘，down 会因生成 2 死锁，up 两种生成都活。
    ROUND2_FATAL = [[2, 4, 8, 2], [8, 32, 16, 2], [16, 2, 64, 16], [4, 16, 128, 32]]
    # 第 3 局（game-ed0c6053…）最后一步的真实棋盘：1 个空位，up 因生成 4 死锁，right 两种生成都活。
    ROUND3_FATAL = [[4, 2, 8, 2], [2, 64, 16, 0], [8, 128, 32, 4], [2, 4, 2, 16]]
    # 第 1 局（game-29e70562…）第 131 步：满盘，up/down 都「生成 2 即死」——不可避免。
    ROUND1_UNAVOIDABLE = [[4, 8, 2, 8], [8, 16, 4, 2], [2, 64, 128, 4], [2, 16, 8, 2]]

    def test_round2_avoidable_death_excludes_the_lethal_direction(self):
        safe, risky = game.analyse_directions(self.ROUND2_FATAL)
        self.assertIn("up", safe)
        self.assertNotIn("down", safe)
        self.assertEqual([d for d, _ in risky], ["down"])
        self.assertIn("spawn 2", dict(risky)["down"])

    def test_round3_avoidable_death_excludes_the_lethal_direction(self):
        # 真实第 3 局终局盘：只有 right 在两种生成下都活；up 与 down 各自会「生成 4 即死」。
        safe, risky = game.analyse_directions(self.ROUND3_FATAL)
        self.assertEqual(safe, ["right"])
        self.assertEqual({d for d, _ in risky}, {"up", "down"})
        self.assertIn("spawn 4", dict(risky)["up"])
        self.assertIn("spawn 4", dict(risky)["down"])

    def test_round1_unavoidable_death_reports_no_safe_direction(self):
        """所有方向都有风险时必须如实报告，不能编造安全选项。"""
        safe, risky = game.analyse_directions(self.ROUND1_UNAVOIDABLE)
        self.assertEqual(safe, [])
        self.assertEqual({d for d, _ in risky}, {"up", "down"})
        for _, why in risky:
            self.assertIn("spawn 2", why)

    def test_spawn_outcomes_covers_every_empty_cell_and_both_values(self):
        board = [[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4]
        after, gain, outcomes = game.spawn_outcomes(board, "left")
        empties = sum(row.count(0) for row in after)
        self.assertEqual(len(outcomes), empties * 2)
        self.assertEqual({spawn for spawn, _, _ in outcomes}, {2, 4})
        expected_cells = sorted((y, x) for y in range(4) for x in range(4) if after[y][x] == 0)
        self.assertEqual(sorted(cell for _, cell, _ in outcomes), sorted(expected_cells * 2))
        self.assertEqual(gain, 0)

    def test_legal_directions_lists_only_board_changing_moves(self):
        self.assertEqual(game.legal_directions(self.ROUND1_UNAVOIDABLE), ["up", "down"])
        stuck = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        self.assertEqual(game.legal_directions(stuck), [])

    def test_analysis_does_not_mutate_the_real_board(self):
        board = [row[:] for row in self.ROUND3_FATAL]
        snapshot = copy.deepcopy(board)
        game.analyse_directions(board)
        self.assertEqual(board, snapshot)


class RunnerSafetyTests(unittest.IsolatedAsyncioTestCase):
    """阶段二：runner 是否真的只把安全方向（+ stop）提供给 Jev。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.browser = FakeBrowser()
        for p in (patch.object(game, "REPORT_DIR", Path(self.tmp.name)),
                  patch.object(game, "Browser2048", return_value=self.browser),
                  patch.dict("os.environ", {"TYPESAFE_API_KEY": "offline-secret"})):
            p.start()
            self.addCleanup(p.stop)

    async def test_runner_offers_only_safe_directions_when_a_safe_one_exists(self):
        # 真实可避免死亡盘：只应把 up + stop 给 Jev，绝不能出现 down。
        self.browser.current = {"id": "t", "state": "playing", "score": 1532, "moveCount": 163,
                                "board": SafetyFilterTests.ROUND2_FATAL}
        seen = []

        def jev(state, question, choices):
            seen.append(dict(choices))
            return {"ok": True, "choice": "stop", "confidence": .5,
                    "probabilities": {k: 1 / len(choices) for k in choices}, "usage": {}}

        with patch.object(game, "jev_choose", side_effect=jev):
            result = await game.run_game(1, 20)
        self.assertEqual(len(seen), 1)
        self.assertEqual(set(seen[0]), {"up", "stop"})
        self.assertNotIn("down", seen[0])
        self.assertEqual(result["reason"], "jev_stop")

    async def test_runner_falls_back_to_risky_directions_when_none_are_safe(self):
        # 不可避免死亡盘：没有安全方向时，如实提供全部合法方向 + stop。
        self.browser.current = {"id": "t", "state": "playing", "score": 1232, "moveCount": 130,
                                "board": SafetyFilterTests.ROUND1_UNAVOIDABLE}
        seen = []

        def jev(state, question, choices):
            seen.append(dict(choices))
            return {"ok": True, "choice": "stop", "confidence": .5,
                    "probabilities": {k: 1 / len(choices) for k in choices}, "usage": {}}

        with patch.object(game, "jev_choose", side_effect=jev):
            result = await game.run_game(1, 20)
        self.assertEqual(set(seen[0]), {"up", "down", "stop"})

    async def test_report_records_safe_and_risky_directions(self):
        self.browser.current = {"id": "t", "state": "playing", "score": 1528, "moveCount": 161,
                                "board": SafetyFilterTests.ROUND3_FATAL}

        def jev(state, question, choices):
            return {"ok": True, "choice": "stop", "confidence": .5,
                    "probabilities": {k: 1 / len(choices) for k in choices}, "usage": {}}

        with patch.object(game, "jev_choose", side_effect=jev):
            result = await game.run_game(1, 20)
        report = json.loads(Path(result["report_path"]).read_text())
        decision = report["decisions"][0]
        self.assertEqual(decision["safe_directions"], ["right"])
        self.assertEqual({r["direction"] for r in decision["risky_directions"]}, {"up", "down"})
        self.assertTrue(decision["filtered_to_safe"])


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        lessons = Path(self.tmp.name) / "lessons.md"
        lessons.write_text("### 下次具体如何选择\nKeep space for merges TEST-LESSON.\n\n"
                           "本局证据只留在文件，不进入请求。\n")
        self.browser = FakeBrowser()
        for p in (patch.object(game, "REPORT_DIR", Path(self.tmp.name)),
                  patch.object(game, "LESSONS_PATH", lessons),
                  patch.object(game, "Browser2048", return_value=self.browser),
                  patch.dict("os.environ", {"TYPESAFE_API_KEY": "offline-secret", "MIE_ENABLE_2048": "1"})):
            p.start()
            self.addCleanup(p.stop)

    async def decide(self, current, choices, lessons=None):
        choice = next(k for k in choices if k != "stop")
        return {"ok": True, "choice": choice, "confidence": .75,
                "probabilities": {k: float(k == choice) for k in choices}, "usage": {"input_tokens": 10}}

    async def test_step_bound_and_report(self):
        with patch.object(game, "_choose", side_effect=self.decide) as choose:
            result = await game.run_game(3, 20)
        self.assertTrue(result["ok"])
        self.assertEqual(result["reason"], "step_limit")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(choose.await_count, 3)
        self.assertEqual(len(self.browser.moves), 3)
        self.assertTrue(self.browser.closed)
        report = json.loads(Path(result["report_path"]).read_text())
        self.assertEqual(len(report["decisions"]), 3)
        self.assertTrue(all(d["verified"] for d in report["decisions"]))
        self.assertNotIn("offline-secret", json.dumps(report))
        self.assertIn("jev_seconds", report["decisions"][0])

    async def test_bad_limits_or_no_key_never_open_browser(self):
        for steps, seconds in ((0, 20), (201, 20), (True, 20), (1, 0), (1, 301)):
            self.assertFalse((await game.run_game(steps, seconds))["ok"])
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": ""}):
            self.assertEqual((await game.run_game(1, 20))["reason"], "not_configured")
        self.assertFalse(self.browser.started)

    async def test_failed_or_abstaining_jev_never_moves(self):
        for answer in ({"ok": False, "error": "http_error"},
                       {"ok": True, "choice": "stop"},
                       {"ok": True, "choice": "up"}):  # up cannot change this board
            with patch.object(game, "_choose", return_value=answer):
                result = await game.run_game(2, 20)
            self.assertEqual(result["steps"], 0)
            self.assertEqual(self.browser.moves, [])
            self.assertTrue(self.browser.closed)

    async def test_read_failure_never_calls_jev(self):
        self.browser.read = AsyncMock(side_effect=game.GameError("board_unavailable"))
        with patch.object(game, "_choose") as choose:
            result = await game.run_game(2, 20)
        choose.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertTrue(self.browser.closed)

    async def test_user_move_while_jev_runs_stops_stale_action(self):
        async def decide(current, choices, lessons=None):
            answer = await self.decide(current, choices, lessons)
            self.browser.current["moveCount"] += 1
            return answer
        with patch.object(game, "_choose", side_effect=decide):
            result = await game.run_game(2, 20)
        self.assertEqual(result["reason"], "board_changed")
        self.assertEqual(self.browser.moves, [])

    async def test_wrong_transition_stops_without_retry(self):
        self.browser.move = AsyncMock(return_value=state())  # Dispatched but page did not change.
        with patch.object(game, "_choose", side_effect=self.decide) as choose:
            result = await game.run_game(4, 20)
        self.assertFalse(result["ok"])
        self.assertEqual(choose.await_count, 1)
        self.browser.move.assert_awaited_once()

    async def test_recorded_lessons_summary_is_exactly_what_was_sent(self):
        """报告记录的摘要必须与实际发给 Jev 的请求内容逐字一致，且经真实组装代码产生。

        这里不改写 load_lessons，也不自己拼接请求：经验摘要从临时测试文件真实读取，
        通过在 jev_choose 边界截获真实请求参数来核对「发送内容」。
        """
        real_summary = game.load_lessons()
        self.assertIsNotNone(real_summary)
        captured = []

        def fake_jev(state, question, choices):
            # 真实请求装配发生在 _choose 内部；这里只截获它最终提交给 Jev 的文本。
            captured.append({"state": state, "question": question, "choices": dict(choices)})
            choice = next(k for k in choices if k != "stop")
            return {"ok": True, "choice": choice, "confidence": .75,
                    "probabilities": {k: float(k == choice) for k in choices},
                    "usage": {"input_tokens": 10}}

        with patch.object(game, "jev_choose", side_effect=fake_jev):
            result = await game.run_game(3, 20)
        self.assertTrue(result["ok"])
        self.assertTrue(result["lessons_loaded"])
        self.assertTrue(result["lessons_sent_to_jev"])
        # 实际发送的请求数等于决策步数，且每一步都带了同一条摘要。
        self.assertEqual(len(captured), 3)
        for request in captured:
            self.assertIn(real_summary, request["question"])
        # 报告里记录的摘要，必须与真实经验文件派生的摘要完全一致。
        self.assertEqual(result["lessons_summary"], real_summary)
        report = json.loads(Path(result["report_path"]).read_text())
        self.assertEqual(report["lessons_summary"], real_summary)
        self.assertTrue(all(d["lessons_sent"] for d in report["decisions"]))

    async def test_lessons_summary_is_derived_not_hardcoded(self):
        """换一份经验文件内容，发送的摘要应随之改变，证明没有固定常量冒充经验。"""
        text = (
            "## 经验 1：测试专用条目\n\n"
            "### 下次具体如何选择\n"
            "Always prefer the corner UNIQUE-MARKER-DIRECTIVE when two moves tie.\n\n"
            "## 经验 2：另一条\n\n"
            "### 下次具体如何选择\n"
            "Never move the largest tile away from its corner SECONDLY.\n"
        )
        summary = game.build_lessons_summary(text)
        self.assertIn("UNIQUE-MARKER-DIRECTIVE", summary)
        self.assertIn("SECONDLY", summary)
        self.assertTrue(summary.startswith(game.LESSONS_HEADER))
        self.assertTrue(summary.endswith(game.LESSONS_FOOTER))
        self.assertNotIn("(3)", summary)

    async def test_lessons_text_reaches_the_real_jev_request_assembly(self):
        """摘要必须经由 _choose -> jev_choose 的真实装配路径，而不是另写的拼接。"""
        captured = {}

        def fake_jev(state, question, choices):
            captured["question"] = question
            captured["choices"] = dict(choices)
            choice = next(k for k in choices if k != "stop")
            return {"ok": True, "choice": choice, "confidence": .9,
                    "probabilities": {k: float(k == choice) for k in choices}, "usage": {}}

        with patch.object(game, "jev_choose", side_effect=fake_jev):
            await game.run_game(1, 20)
        self.assertIn(game.LESSONS_HEADER, captured["question"])
        self.assertIn(game.LESSONS_FOOTER, captured["question"])
        # 选项说明与 stop 都来自同一装配过程。
        self.assertIn("stop", captured["choices"])

    async def test_missing_lessons_file_still_plays_and_records_none(self):
        """经验文件缺失或为空时，整局照常进行，报告如实记录未加载。"""
        for loader in ("missing", "empty"):
            with self.subTest(loader=loader):
                self.browser.current = state()
                with patch.object(game, "load_lessons", return_value=None), \
                     patch.object(game, "_choose", side_effect=self.decide) as choose:
                    result = await game.run_game(2, 20)
                self.assertTrue(result["ok"])
                self.assertEqual(result["reason"], "step_limit")
                self.assertEqual(result["steps"], 2)
                self.assertFalse(result["lessons_loaded"])
                self.assertIsNone(result["lessons_summary"])
                self.assertFalse(result["lessons_sent_to_jev"])
                self.assertEqual(choose.await_count, 2)
                report = json.loads(Path(result["report_path"]).read_text())
                self.assertFalse(any(d["lessons_sent"] for d in report["decisions"]))

    def test_load_lessons_tolerates_missing_empty_and_directory(self):
        """load_lessons 对缺失、空白、目录、不可读都要返回 None 而不抛异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                "missing": root / "nope.md",
                "empty": root / "empty.md",
                "blank": root / "blank.md",
                "directory": root / "dir",
            }
            cases["empty"].write_text("")
            cases["blank"].write_text("   \n\t ")
            cases["directory"].mkdir()
            for name, target in cases.items():
                with self.subTest(case=name), patch.object(game, "LESSONS_PATH", target):
                    self.assertIsNone(game.load_lessons())
            # 有真实条目时派生出摘要；没有「下次具体如何选择」条目时如实返回 None。
            real = root / "real.md"
            real.write_text("### 下次具体如何选择\nkeep the largest tile in a corner TIMEout\n")
            with patch.object(game, "LESSONS_PATH", real):
                summary = game.load_lessons()
            self.assertIsNotNone(summary)
            self.assertIn("TIMEout", summary)
            no_directive = root / "none.md"
            no_directive.write_text("## 只有标题，没有可执行条目\n")
            with patch.object(game, "LESSONS_PATH", no_directive):
                self.assertIsNone(game.load_lessons())

    async def test_terminal_board_does_not_call_jev(self):
        self.browser.current["state"] = "gameWon"
        self.browser.current["board"][0][0] = 2048
        with patch.object(game, "_choose") as choose:
            result = await game.run_game(2, 20)
        self.assertEqual(result["reason"], "game_won")
        choose.assert_not_called()

    async def test_cancel_during_jev_closes_session_without_late_move(self):
        entered = asyncio.Event()
        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(game, "_choose", side_effect=blocked):
            task = asyncio.create_task(game.run_game(20, 30))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.browser.moves, [])
        self.assertTrue(self.browser.closed)
        report = json.loads(next(Path(self.tmp.name).glob("*.json")).read_text())
        self.assertEqual(report["reason"], "cancelled")

    async def test_cancel_during_real_sync_wrapper_ignores_late_http_result(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def delayed_jev(*args):
            entered.set()
            release.wait(3)
            finished.set()
            return {"ok": True, "choice": "left", "confidence": .9}
        with patch.object(game, "jev_choose", side_effect=delayed_jev):
            task = asyncio.create_task(game.run_game(20, 30))
            try:
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(entered.is_set())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(self.browser.closed)
            finally:
                release.set()
            for _ in range(100):
                if finished.is_set():
                    break
                await asyncio.sleep(.01)
        self.assertEqual(self.browser.moves, [])

    async def test_time_limit_stops_waiting_and_closes(self):
        async def blocked(*args):
            await asyncio.sleep(5)
        with patch.object(game, "_choose", side_effect=blocked):
            result = await game.run_game(20, 1)
        self.assertEqual(result["reason"], "time_limit")
        self.assertEqual(self.browser.moves, [])
        self.assertTrue(self.browser.closed)

    async def test_actual_agent_approves_whole_loop_once_and_rejects_without_execution(self):
        from app.agent import build_agent
        from tests.test_tui import ScriptedModel
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.types import Command
        for verdict in ("reject", "approve"):
            self.browser.started = False
            model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[{
                "name": "play_2048", "id": "g1", "args": {"max_steps": 2, "max_seconds": 20},
            }]), AIMessage(content="结束")])
            with patch.dict("os.environ", {"AGENTSEEK_MODEL": "offline", "OPENAI_API_KEY": "offline",
                                           "OPENAI_API_BASE": "https://example.invalid"}), patch("app.agent.ChatOpenAI", return_value=model):
                graph = build_agent()
            config = {"configurable": {"thread_id": verdict}}
            with patch.object(game, "_choose", side_effect=self.decide) as choose:
                pending = await graph.ainvoke({"messages": [{"role": "user", "content": "玩两步2048"}]}, config)
                self.assertFalse(self.browser.started)
                pause = pending["__interrupt__"][0]
                self.assertEqual(pause.value["action_requests"][0]["name"], "play_2048")
                output = await graph.ainvoke(Command(resume={pause.id: {"decisions": [{"type": verdict}]}}), config)
            self.assertNotIn("__interrupt__", output)
            self.assertEqual(choose.await_count, 2 if verdict == "approve" else 0)
            if verdict == "approve":
                message = next(m for m in output["messages"] if isinstance(m, ToolMessage) and m.name == "play_2048")
                self.assertEqual(json.loads(message.content)["steps"], 2)

    async def test_tui_escape_cancels_the_actual_game_tool(self):
        from app.agent import build_agent
        from app.tui import MieApp, Composer
        from tests.test_tui import ScriptedModel
        from langchain_core.messages import AIMessage
        entered = asyncio.Event()
        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()
        model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[{
            "name": "play_2048", "id": "g2", "args": {"max_steps": 20, "max_seconds": 30},
        }]), AIMessage(content="结束")])
        with patch.dict("os.environ", {"AGENTSEEK_MODEL": "offline", "OPENAI_API_KEY": "offline",
                                       "OPENAI_API_BASE": "https://example.invalid"}), patch("app.agent.ChatOpenAI", return_value=model):
            graph = build_agent()
        with patch.object(game, "_choose", side_effect=blocked):
            app = MieApp(graph)
            async with app.run_test() as pilot:
                await pilot.press("tab")
                app.query_one(Composer).load_text("玩2048")
                await pilot.press("enter")
                await asyncio.wait_for(entered.wait(), 3)
                await pilot.press("escape")
                for _ in range(100):
                    if not app.busy:
                        break
                    await pilot.pause(.01)
                self.assertFalse(app.busy)
                self.assertTrue(self.browser.closed)
                self.assertEqual(self.browser.moves, [])
                self.assertEqual(app.phase, "已停止")


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_move_returns_readback_in_one_operation_and_rejects_stale_board(self):
        browser = game.Browser2048()
        browser.session, browser.tab = "owned", 123
        after = state([[4, 2, 0, 0], [0]*4, [0]*4, [0]*4])
        after.update(moveCount=1, score=4)
        browser.command = AsyncMock(return_value={"ok": True, "value": after})
        self.assertEqual(await browser.move("left", state()), after)
        self.assertEqual(browser.command.await_count, 1)
        self.assertEqual(browser.command.call_args.args[0], "evaluate")
        browser.command.return_value = {"ok": True, "value": {"error": "board_changed"}}
        with self.assertRaisesRegex(game.GameError, "board_changed"):
            await browser.move("left", state())

    async def test_exit_zero_with_javascript_failure_is_not_success(self):
        process = AsyncMock()
        process.returncode = 0
        process.communicate.return_value = (b'{"ok":false,"exception":"private data"}', b'')
        with patch("asyncio.create_subprocess_exec", return_value=process) as spawn:
            with self.assertRaisesRegex(game.GameError, "browser_error"):
                await game.Browser2048().command("evaluate", "test")
        self.assertNotIn("TYPESAFE_API_KEY", spawn.call_args.kwargs["env"])

    async def test_ambiguous_browser_connection_does_not_create_session(self):
        for browsers in ([], [{"instance_id": "a"}, {"instance_id": "b"}]):
            browser = game.Browser2048()
            browser.command = AsyncMock(return_value={"browsers": browsers})
            with self.assertRaisesRegex(game.GameError, "connect_one_browser"):
                await browser.start()
            browser.command.assert_awaited_once_with("status")


if __name__ == "__main__":
    unittest.main()
