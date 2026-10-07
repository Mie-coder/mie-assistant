import asyncio
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.features.game2048 import runtime as game
from app.features.game2048 import solver
from tests.test_game2048 import FakeBrowser, state


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver.build()
        cls.native = solver.load_solver()

    def test_native_transitions_match_independent_python_for_all_directions(self):
        rng = random.Random(2026)
        for _ in range(500):
            board = [[rng.choice([0, 2, 4, 8, 16, 32, 64, 128, 256, 512]) for _ in range(4)] for _ in range(4)]
            packed = solver.encode(board)
            for direction, index in solver.MOVES.items():
                native = self.native.execute_move(index, packed)
                expected, _ = game.slide(board, direction)
                self.assertEqual(native, solver.encode(expected), (board, direction))

    def test_search_respects_candidates_and_has_finite_scores(self):
        board = [[2, 2, 4, 8], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
        result = solver.recommend(board, ["left"], budget_ms=10)
        self.assertEqual(result["direction"], "left")
        self.assertEqual(set(result["scores"]), {"left"})
        self.assertGreaterEqual(result["completed_depth"], 0)

    def test_reject_unrepresentable_boards_and_noop_candidates(self):
        for tile in (True, 1, 3, -2, 65536):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                solver.encode([[tile, 0, 0, 0], [0]*4, [0]*4, [0]*4])
        with self.assertRaises(RuntimeError):
            solver.recommend([[2, 0, 0, 0], [0]*4, [0]*4, [0]*4], ["up"])


class HybridTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.browser = FakeBrowser()
        self.patches = [patch.object(game, "Browser2048", return_value=self.browser),
                        patch.object(game, "REPORT_DIR", Path(self.temp.name)),
                        patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-only"}),
                        patch.object(game, "load_solver")]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def search(*args):
        return {"direction": "right", "scores": {"left": 10., "right": 20., "down": 5.},
                "completed_depth": 3, "seconds": .001}

    async def test_concurrent_search_corrects_jev_without_rewriting_its_result(self):
        entered = threading.Event()

        def search(*args):
            entered.set()
            return self.search()

        async def jev(*args):
            self.assertTrue(await asyncio.to_thread(entered.wait, .5))
            return {"ok": True, "choice": "left", "confidence": .5}

        with patch.object(game, "recommend", side_effect=search), patch.object(game, "_choose", side_effect=jev):
            result = await game.run_game(1, 20, search_assist=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.browser.moves, ["right"])
        self.assertEqual(result["overrides"], 1)
        decision = json.loads(Path(result["report_path"]).read_text())["decisions"][0]
        self.assertEqual(decision["result"]["choice"], "left")
        self.assertEqual(decision["executed_direction"], "right")
        self.assertTrue(decision["verified"])

    async def test_jev_stop_and_errors_are_never_overridden(self):
        for response in ({"ok": True, "choice": "stop"}, {"ok": False, "error": "http_error"}):
            with patch.object(game, "recommend", side_effect=self.search), patch.object(game, "_choose", return_value=response):
                result = await game.run_game(2, 20, search_assist=True)
            self.assertEqual(result["steps"], 0)
            self.assertEqual(self.browser.moves, [])

    async def test_solver_failure_stops_without_using_jev_alone(self):
        with patch.object(game, "recommend", side_effect=RuntimeError("failed")), patch.object(
                game, "_choose", return_value={"ok": True, "choice": "left"}):
            result = await game.run_game(2, 20, search_assist=True)
        self.assertEqual(result["reason"], "solver_error")
        self.assertEqual(self.browser.moves, [])

    async def test_fake_win_marker_is_not_success(self):
        self.browser.current["state"] = "gameWon"
        with patch.object(game, "_choose") as choose:
            result = await game.run_game(2, 20)
        self.assertEqual(result["reason"], "win_not_verified")
        self.assertFalse(result["ok"])
        choose.assert_not_called()

    async def test_2048_stops_even_before_won_marker_updates(self):
        self.browser.current["board"][0][0] = 2048
        with patch.object(game, "_choose") as choose:
            result = await game.run_game(2, 20)
        self.assertEqual(result["reason"], "game_won")
        choose.assert_not_called()

    async def test_continuation_reuses_own_browser_and_rejects_changed_board(self):
        self.browser.session, self.browser.tab = "owned-test", 123
        self.browser.start = AsyncMock()
        async def choose(current, choices, lessons=None):
            return {"ok": True, "choice": next(d for d in choices if d != "stop")}
        with patch.object(game, "_choose", side_effect=choose):
            first = await game.run_game(1, 20, keep_page_open=True)
            self.assertFalse(self.browser.closed)
            second = await game.run_game(1, 20, keep_page_open=True, continuation_id=first["continuation_id"])
        self.browser.start.assert_awaited_once()
        self.assertEqual(second["steps"], 1)
        self.browser.current["moveCount"] += 1
        third = await game.run_game(1, 20, continuation_id=second["continuation_id"])
        self.assertEqual(third["reason"], "board_changed")
        self.assertTrue(self.browser.closed)
        reused = await game.run_game(1, 20, continuation_id=first["continuation_id"])
        self.assertEqual(reused["reason"], "invalid_continuation")

    async def test_cancelled_hybrid_never_executes_delayed_search(self):
        entered = asyncio.Event()
        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(game, "recommend", side_effect=self.search), patch.object(game, "_choose", side_effect=blocked):
            task = asyncio.create_task(game.run_game(2, 20, search_assist=True))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.browser.moves, [])
        self.assertTrue(self.browser.closed)


class RestartTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_game_requires_verified_terminal_board_and_button_click(self):
        browser = game.Browser2048()
        browser.session, browser.tab = "owned-test", 123
        before = state([[2,4,8,16], [4,8,2,8], [16,32,256,16], [2,8,16,32]])
        before["state"] = "gameOver"
        fresh = state()
        fresh["id"] = "new-game"
        browser.read = AsyncMock(return_value=before)
        # A restored gameOver store precedes the terminal overlay's mount.
        browser.command = AsyncMock(side_effect=[{"text": '@e3 button "N or R New Game"'},
                                                {"text": '@e2 button "Play Again"'}, {"ok": True, "value": fresh}])
        self.assertEqual(await browser.new_game(before), fresh)
        action = browser.command.call_args_list[2].args
        self.assertEqual(action[0], "evaluate")
        self.assertIn('"restart":true', action[1])
        self.assertIn('buttons[0].click()', action[1])
        browser.command.reset_mock()
        with self.assertRaises(game.GameError):
            await browser.new_game(state())
        browser.command.assert_not_called()
