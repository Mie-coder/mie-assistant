#include <chrono>
#include <cmath>
#include <limits>

namespace {
using Clock = std::chrono::steady_clock;
struct BudgetExpired {};
thread_local Clock::time_point deadline;
thread_local unsigned visits;
void check_budget() {
    if ((++visits & 255) == 0 && Clock::now() >= deadline) throw BudgetExpired{};
}
}
#define MIE_SEARCH_CHECK() check_budget()
#include "2048.cpp"

// Iterative deepening: compare all allowed moves at the SAME completed depth.
// A partial iteration is discarded, never compared to shallower scores.
extern "C" DLL_PUBLIC int mie_scores(board_t board, unsigned mask, int budget_ms, float *out) {
    if (!board || !(mask & 15) || budget_ms < 1 || budget_ms > 200) return -1;
    deadline = Clock::now() + std::chrono::milliseconds(budget_ms);
    visits = 0;
    int completed = -1;
    try {
        const int limit = std::max(3, std::min(8, count_distinct_tiles(board) - 2));
        for (int depth = 0; depth <= limit; ++depth) {
            float iteration[4];
            for (int move = 0; move < 4; ++move) {
                iteration[move] = -std::numeric_limits<float>::infinity();
                if (!(mask & (1u << move)) || execute_move(move, board) == board) continue;
                if (Clock::now() >= deadline) throw BudgetExpired{};
                eval_state state;
                state.depth_limit = depth;
                iteration[move] = _score_toplevel_move(state, board, move);
            }
            std::copy(iteration, iteration + 4, out);
            completed = depth;
        }
    } catch (const BudgetExpired &) {
        // Keep the last fully scored iteration.
    } catch (...) {
        return -1;
    }
    return completed;
}
