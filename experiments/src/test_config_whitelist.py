"""Verify the platform-config flagos_whitelist behaves as a vendor default.

Single-process: import flag_gems/vllm_fl once, then flip env vars between
cases. Re-importing per case costs ~2 min each, so keep it in one process.

Cases:
  1. no env                 -> config whitelist wins (3 ops)
  2. WHITELIST env          -> env wins over config
  3. BLACKLIST env          -> env blacklist wins, config whitelist ignored
  4. BLACKLIST_APPEND only  -> config whitelist still wins (append ignored)
  5. config -> verify the actual dispatch decision for listed vs unlisted ops
"""

import os

from vllm_fl.dispatch.config import get_flagos_whitelist
from vllm_fl.utils import get_flag_gems_whitelist_blacklist, use_flaggems_op

ENV_KEYS = (
    "VLLM_FL_FLAGOS_WHITELIST",
    "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
)

cases = [
    ("config default (no env)", {}, "config", 3),
    ("env whitelist overrides", {"VLLM_FL_FLAGOS_WHITELIST": "mm,rms_norm"}, "whitelist", ["mm", "rms_norm"]),
    ("env blacklist overrides", {"VLLM_FL_FLAGOS_BLACKLIST": "rms_norm"}, "blacklist", None),
    ("append ignored under config wl", {"VLLM_FL_FLAGOS_BLACKLIST_APPEND": "zeros"}, "config", 3),
]

print("=== raw config read ===")
print("  get_flagos_whitelist() =", get_flagos_whitelist())
print()

failures = 0
for name, env, expect_kind, expect_val in cases:
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)

    wl, bl = get_flag_gems_whitelist_blacklist()
    if expect_kind == "config":
        ok = wl is not None and len(wl) == expect_val and bl is None
    elif expect_kind == "whitelist":
        ok = wl == expect_val and bl is None
    else:
        ok = wl is None and bl == ["rms_norm"]

    print(f"[{name}]")
    print(f"  env       : {env or '<none>'}")
    print(f"  whitelist : {wl}")
    print(f"  blacklist : {bl}")
    print(f"  -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        failures += 1

# Case 5: the decision actually used by the dispatch registry.
for k in ENV_KEYS:
    os.environ.pop(k, None)
print("\n[dispatch decisions under config whitelist]")
for op in ("silu_and_mul", "rms_norm", "rotary_embedding", "mm", "zeros", "linear"):
    d = use_flaggems_op(op)
    print(f"  use_flaggems_op({op:18s}) = {d}")

expected = {
    "silu_and_mul": True,
    "rms_norm": True,
    "rotary_embedding": True,
    "mm": False,
    "zeros": False,
    "linear": False,
}
bad = [op for op, want in expected.items() if use_flaggems_op(op) != want]
if bad:
    print(f"  !! wrong decision for: {bad}")
    failures += 1
else:
    print("  -> OK (3 listed ops on FlagGems, everything else falls back to vendor)")

print(f"\n{'ALL OK' if failures == 0 else f'{failures} FAILURE(S)'}")
raise SystemExit(1 if failures else 0)
