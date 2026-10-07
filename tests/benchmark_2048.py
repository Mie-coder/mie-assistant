"""Offline seeded smoke benchmark. No browser, Jev, or real-game randomness."""
import json
import random
from pathlib import Path
from statistics import mean

from app.features.game2048.runtime import analyse_directions, slide
from app.features.game2048.solver import recommend


def run(seed):
    rng = random.Random(seed)
    board = [[0] * 4 for _ in range(4)]
    score, times, steps = 0, [], 0

    def spawn():
        y, x = rng.choice([(y, x) for y in range(4) for x in range(4) if board[y][x] == 0])
        board[y][x] = 2 if rng.random() < .9 else 4

    spawn()
    spawn()
    while max(map(max, board)) < 2048 and steps < 2000:
        safe, risky = analyse_directions(board)
        offered = safe or [d for d, _ in risky]
        if not offered:
            break
        result = recommend(board, offered)
        board, gain = slide(board, result["direction"])
        score += gain
        spawn()
        steps += 1
        times.append(result["seconds"])
    return {"seed": seed, "won": max(map(max, board)) >= 2048, "max_tile": max(map(max, board)),
            "score": score, "steps": steps, "search_mean_ms": mean(times) * 1000,
            "search_max_ms": max(times) * 1000, "board": board}


if __name__ == "__main__":
    results = []
    output = Path("workspace/reports/2048/search-benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for seed in (19, 2048, 2026):
        result = run(seed)
        results.append(result)
        output.write_text(json.dumps({"mode": "offline_expectimax_only", "games": results}, indent=2))
        print(json.dumps(result), flush=True)
