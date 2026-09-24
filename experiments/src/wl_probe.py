#!/usr/bin/env python
"""Report the resolved FlagGems routing for a set of representative ops.

Usage: wl_probe.py [label]
Reads the live platform config (metax.yaml) and env, so it reflects whatever
arm is currently installed.
"""

import os
import sys

LABEL = sys.argv[1] if len(sys.argv) > 1 else "?"

for k in ("VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST"):
    os.environ.pop(k, None)

from vllm_fl.utils import get_flag_gems_whitelist_blacklist, use_flaggems_op

OPS = ["mm", "linear", "bmm", "addmm", "silu_and_mul", "rms_norm",
       "rotary_embedding", "scaled_dot_product_attention", "flash_attn",
       "softmax", "layer_norm", "copy", "cat", "mul", "add", "topk"]

wl, bl = get_flag_gems_whitelist_blacklist()
y = [o for o in OPS if use_flaggems_op(o)]
n = [o for o in OPS if not use_flaggems_op(o)]
print(f"[{LABEL}] whitelist={wl} blacklist={bl}")
print(f"[{LABEL}] FlagGems: {', '.join(y) if y else '(none)'}")
print(f"[{LABEL}] native  : {', '.join(n) if n else '(none)'}")
print(f"[{LABEL}] flagoss_count={len(y)}/{len(OPS)}")
