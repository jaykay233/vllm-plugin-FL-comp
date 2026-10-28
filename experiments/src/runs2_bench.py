#!/usr/bin/env python3
"""Run benchmark_throughput_serve.py with a configurable number of runs.

The shared harness hardcodes `RUNS = 4` / `SKIP_FIRST = 1` as module constants.
Once an arm's first two valid runs agree to ~0.1% (as arm A's did: 11973.65 and
12013.55), the remaining two runs only buy confirmation we do not need, and each
one costs ~108 s of GPU time.

Sets the constants before `main()` so the shared file stays untouched, and sets
SKIP_FIRST to 0 because there is no warmup run to discard at RUNS=2 -- the first
run is already steady state (the server is warmed during graph capture).

Usage:
    runs2_bench.py <original argv...>        # RUNS=2, SKIP_FIRST=0
    RUNS=3 runs2_bench.py <argv...>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/workspace/vllm-plugin-FL/benchmarks")

import benchmark_throughput_serve as b  # noqa: E402


def main() -> int:
    runs = int(os.environ.get("RUNS", "2"))
    skip_first = int(os.environ.get("SKIP_FIRST", "0"))
    b.RUNS = runs
    b.SKIP_FIRST = skip_first
    print(f"[runs2] RUNS={runs} SKIP_FIRST={skip_first}", flush=True)
    b.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
