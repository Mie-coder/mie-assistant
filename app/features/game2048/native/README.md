# 2048 search core

Vendored from https://github.com/nneonneo/2048-ai at commit
`41e298f4571a9505e421e3a19af7a1cb372a368c` (MIT; see LICENSE).

`2048.h` and `platdefs.h` are unchanged. `2048.cpp` has one optional
`MIE_SEARCH_CHECK` hook at the start of `score_tilechoose_node` to enforce the
local time budget. `config.h` and `adapter.cpp` are MIE integration code.
The adapter compares completed Expectimax iterations, limits each search to
80 ms by default, and emits no stdout. Deadline polling is cooperative (not a
hard real-time guarantee). Spawn probabilities use the core's standard 90% 2 /
10% 4 assumption; it never reads the website RNG.

Only calculation runs here. Browser controllers from upstream are not included.
The native 4-bit encoding supports tile values up to 32768; MIE stops at 2048.

Build from `mie_assistant/`: `.venv/bin/python -m app.features.game2048.solver --build`.
Requires a local C++11 compiler. The source-hashed library is a local cache in
`workspace/.cache/2048/`, not a downloaded executable.
