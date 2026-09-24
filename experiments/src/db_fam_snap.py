"""Snapshot the per-family autotune table counts of the FlagGems cache DB.

Usage: db_fam_snap.py <db> <out.txt>
Output lines: <family>\ttables=<n>\trows=<n>
"""

import sqlite3
import sys
from collections import Counter

FAMILIES = ("mm_kernel_nt", "mm_kernel_splitk", "mm_kernel_nn", "mm_kernel_general",
            "linear_kernel")


def family_of(table: str):
    for f in FAMILIES:
        if table.startswith(f):
            return f
    return None


def main() -> int:
    db, out = sys.argv[1], sys.argv[2]
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tabs = [r[0] for r in conn.execute(
        "select name from sqlite_master where type='table'")]
    tables, rows = Counter(), Counter()
    for t in tabs:
        f = family_of(t)
        if not f:
            continue
        tables[f] += 1
        try:
            rows[f] += conn.execute(f'select count(*) from "{t}"').fetchone()[0]
        except Exception:
            pass
    with open(out, "w") as fh:
        for f in sorted(set(tables) | set(rows)):
            fh.write(f"{f}\ttables={tables[f]}\trows={rows[f]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
