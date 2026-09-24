"""Add the strategy= the framework already declares but the MetaX mm decorators omit.

runtime/common.py DEFAULT_STRATEGIES says mm/mm_nn/mm_nt/mm_splitk/gemv should
bucket their keys with align32, but that table is only consumed in
TuningMode.EXPANDED. These decorators run in TuningMode.DEFAULT and pass no
strategy=, so they fall back to the identity strategy and key the autotuner on
the raw runtime M -- which changes every decode step and produces a hot-path
autotune storm (see vllm-plugin-FL experiments/mm_metax_rootcause.md section 10).

This inserts strategy=[...] after each key=[...] line, using exactly the value
DEFAULT_STRATEGIES declares. Undeclared kernels in the same family
(gemv_k_parallel, mm_splitk_two_step) mirror their declared sibling.

Idempotent: skips any block that already has strategy=.
"""

import re
import sys

# expand_op_name -> strategy (must match key length)
DECLARED = {
    "mm": ["align32"] * 5,
    "mm_nn": ["align32"] * 3,
    "mm_nt": ["align32"] * 3,
    "mm_splitk": ["align32"] * 5,
    "gemv": ["align32", "align32", "align32", "default"],
    # not in DEFAULT_STRATEGIES; mirror the declared sibling for the same key shape
    "gemv_k_parallel": ["align32", "align32", "align32", "default"],
    "mm_splitk_two_step": ["align32"] * 5,
}


def patch(path: str) -> int:
    lines = open(path).read().splitlines(keepends=True)
    out, i, added = [], 0, 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("@libtuner("):
            # collect the whole decorator block
            block, depth, j = [], 0, i
            while j < len(lines):
                block.append(lines[j])
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and j > i:
                    break
                j += 1
            text = "".join(block)
            has_strategy = re.search(r"^\s*strategy=", text, re.M) is not None
            key_line = re.search(r"^\s*key=\[([^\]]*)\]\s*,?\s*$", text, re.M)
            op = re.search(r'flagtune_expand_op_name="([^"]+)"', text)
            opname = op.group(1) if op else None
            strategy = DECLARED.get(opname)

            if has_strategy or key_line is None or strategy is None:
                out.extend(block)
                i = j + 1
                if not has_strategy and strategy is None:
                    print(f"  skip   {path.split('/')[-1]} line {i}: "
                          f"no declared strategy for op={opname!r}")
                continue

            nkey = len([x for x in key_line.group(1).split(",") if x.strip()])
            if nkey != len(strategy):
                print(f"  ERROR  op={opname} keylen={nkey} != strategy len={len(strategy)}")
                sys.exit(1)

            # insert right after the key line
            strats = ", ".join(f'"{s}"' for s in strategy)
            kline_idx = None
            for idx, bl in enumerate(block):
                if re.match(r"^\s*key=\[", bl):
                    kline_idx = idx
                    break
            block.insert(kline_idx + 1,
                         f'    strategy=[{strats}],\n')
            out.extend(block)
            added += 1
            print(f"  patch  {path.split('/')[-1]} op={opname:20s} strategy=[{strats}]")
            i = j + 1
            continue
        out.append(line)
        i += 1

    if added:
        open(path, "w").write("".join(out))
    return added


total = 0
for p in sys.argv[1:]:
    print(f"=== {p}")
    total += patch(p)
print(f"\n共修改 {total} 个 decorator")
