"""Point the M dimension of the MetaX mm/gemv tuners at align32_geometric.

M is always the first key element in these decorators, so this replaces the
first entry of each strategy=[...] list. Static dimensions (N, K, strides) keep
align32, which is what DEFAULT_STRATEGIES declares for them and is correct for
values that do not vary at runtime.

Idempotent: skips blocks already using align32_geometric.
"""

import re
import sys

IMPORT_LINE = (
    "from flag_gems.runtime.backend._metax.tuner_strategies import "
    "align32_geometric  # noqa: F401  (registers the strategy)\n"
)
ANCHOR = "from flag_gems.utils.device_info import get_l2_cache_size, get_sm_count\n"


def main() -> int:
    path = sys.argv[1]
    src = open(path).read()
    lines = src.splitlines(keepends=True)

    # 1) ensure the strategy module is imported before the decorators run
    added_import = False
    if "tuner_strategies import" not in src:
        for i, ln in enumerate(lines):
            if ln == ANCHOR:
                lines.insert(i + 1, IMPORT_LINE)
                added_import = True
                break
        else:
            print("ERROR: anchor import line not found")
            return 1

    # 2) rewrite the first entry of every strategy=[...] inside a @libtuner block
    out, i, changed = [], 0, 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("@libtuner("):
            block, depth, j = [], 0, i
            while j < len(lines):
                block.append(lines[j])
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and j > i:
                    break
                j += 1
            text = "".join(block)
            m = re.search(r'strategy=\[([^\]]*)\]', text)
            if m and '"align32_geometric"' not in text:
                parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
                parts[0] = '"align32_geometric"'
                new = "strategy=[" + ", ".join(parts) + "]"
                text = text[:m.start()] + new + text[m.end():]
                block = text.splitlines(keepends=True)
                changed += 1
            out.extend(block)
            i = j + 1
            continue
        out.append(ln)
        i += 1

    open(path, "w").write("".join(out))
    print(f"  import added: {added_import}")
    print(f"  strategy blocks updated: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
