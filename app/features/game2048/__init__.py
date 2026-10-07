"""2048 的轻量工具入口；批准调用后才加载棋盘、浏览器与搜索实现。"""
import asyncio

from langchain_core.tools import StructuredTool

from app.features import game2048_enabled


def _load_runner():
    from .runtime import run_game

    return run_game


async def _play_async(max_steps: int = 50, max_seconds: int = 120, search_assist: bool = False,
                      new_game: bool = False, keep_page_open: bool = False, continuation_id: str | None = None) -> dict:
    if not game2048_enabled():
        return {"ok": False, "reason": "feature_disabled",
                "message": "2048 功能已关闭。设置 MIE_ENABLE_2048=1 并重启后可按需调用；不要绕过开关。"}
    try:
        run_game = _load_runner()
    except (ImportError, OSError):
        return {"ok": False, "reason": "feature_unavailable",
                "message": "2048 模块或资源加载失败；停止本轮，不使用其他方式继续操作。"}
    from app.jev import jev_session

    with jev_session():
        return await run_game(max_steps, max_seconds, search_assist, new_game, keep_page_open, continuation_id)


def _play_sync(max_steps: int = 50, max_seconds: int = 120, search_assist: bool = False,
               new_game: bool = False, keep_page_open: bool = False, continuation_id: str | None = None) -> dict:
    return asyncio.run(_play_async(max_steps, max_seconds, search_assist, new_game, keep_page_open, continuation_id))


play_2048 = StructuredTool.from_function(
    func=_play_sync, coroutine=_play_async, name="play_2048",
    description=(
        "在用户要求玩或续玩 2048 时使用。调用前读取 /skills/game2048/SKILL.md。"
        "审批后按需加载；遵守用户步数、时间和累计预算，未经授权不启用算法辅助或重开。"
        "功能关闭或调用失败时停止，不用 Shell 绕过。"
    ),
)
