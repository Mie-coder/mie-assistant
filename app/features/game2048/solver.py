"""Bounded, native Expectimax assistance; no browser or network access."""
import ctypes
import hashlib
import math
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from time import monotonic

from app.paths import WORKSPACE_DIR

SOURCE = Path(__file__).with_name("native")
CACHE = WORKSPACE_DIR / ".cache" / "2048"
MOVES = {"up": 0, "down": 1, "left": 2, "right": 3}
_library = None
_lock = threading.Lock()


def library_path():
    digest = hashlib.sha256()
    for name in ("2048.cpp", "2048.h", "platdefs.h", "config.h", "adapter.cpp"):
        digest.update((SOURCE / name).read_bytes())
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return CACHE / (digest.hexdigest()[:16] + suffix)


def build():
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("solver_compiler_missing")
    target = library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler, "-std=c++11", "-O3", "-fPIC", "-shared",
                    str(SOURCE / "adapter.cpp"), "-o", str(target)], check=True, timeout=60)
    return target


def load_solver():
    global _library
    with _lock:
        if _library is None:
            path = library_path()
            if not path.is_file():
                raise RuntimeError("solver_not_built")
            lib = ctypes.CDLL(str(path))  # CDLL releases the GIL during search.
            lib.init_tables.argtypes = []
            lib.init_tables.restype = None
            lib.execute_move.argtypes = [ctypes.c_int, ctypes.c_uint64]
            lib.execute_move.restype = ctypes.c_uint64
            lib.mie_scores.argtypes = [ctypes.c_uint64, ctypes.c_uint, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float)]
            lib.mie_scores.restype = ctypes.c_int
            lib.init_tables()
            _library = lib
        return _library


def encode(board):
    if (not isinstance(board, list) or len(board) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in board)):
        raise ValueError("solver_invalid_board")
    result = 0
    for index, tile in enumerate(n for row in board for n in row):
        if type(tile) is not int or tile < 0 or tile > 32768 or (tile and (tile < 2 or tile & (tile - 1))):
            raise ValueError("solver_invalid_board")
        result |= (tile.bit_length() - 1 if tile else 0) << (4 * index)
    if not result:
        raise ValueError("solver_empty_board")
    return result


def recommend(board, offered, budget_ms=80):
    started = monotonic()
    if not offered or any(d not in MOVES for d in offered):
        raise ValueError("solver_invalid_choices")
    if type(budget_ms) is not int or not 1 <= budget_ms <= 200:
        raise ValueError("solver_invalid_budget")
    values = (ctypes.c_float * 4)()
    depth = load_solver().mie_scores(encode(board), sum(1 << MOVES[d] for d in set(offered)), budget_ms, values)
    scores = {d: float(values[MOVES[d]]) for d in offered}
    if depth < 0 or not all(math.isfinite(v) for v in scores.values()):
        raise RuntimeError("solver_search_failed")
    return {"direction": max(scores, key=scores.get), "scores": scores,
            "completed_depth": depth, "budget_ms": budget_ms,
            "seconds": round(monotonic() - started, 6)}


if __name__ == "__main__":
    if sys.argv[1:] != ["--build"]:
        raise SystemExit("Usage: python -m app.features.game2048.solver --build")
    print(build())
