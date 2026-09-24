"""Definitive canary: does FlagGems' mm autotuner run under the config whitelist?

Counting TABLES is the wrong metric (tables are per-kernel, rows are per-shape).
An earlier attempt counted tables and was therefore insensitive. Count ROWS.

If the whitelist is in effect, `mm` is not dispatched to FlagGems, the runtime
Triton autotuner never fires, and row counts stay flat while we feed brand-new
prefill M values.
"""

import json
import sqlite3
import time
import urllib.request

DB = "/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db"
URL = "http://127.0.0.1:9031/v1/completions"


def mm_rows():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tabs = [
        r[0]
        for r in c.execute("select name from sqlite_master where type='table'")
    ]
    total = 0
    per = {}
    for t in tabs:
        if not t.startswith("mm_kernel"):
            continue
        n = c.execute(f'select count(*) from "{t}"').fetchone()[0]
        per[t[:40]] = n
        total += n
    c.close()
    return total, len(per), per


def send(words, ntok=4):
    base = "the quick brown fox jumps over a lazy dog and then "
    prompt = (base * (words // 10 + 1))[: words * 6]
    body = json.dumps(
        {"model": "minicpm", "prompt": prompt, "max_tokens": ntok, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


before, ntabs, _ = mm_rows()
print(f"  BEFORE mm_rows={before} across {ntabs} mm tables", flush=True)

t0 = time.time()
for words in (2003, 1291, 1601, 1801):
    try:
        out = send(words)
        pt = out.get("usage", {}).get("prompt_tokens")
        print(f"  sent words={words} prompt_tokens={pt}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  sent words={words} FAILED {exc}", flush=True)
elapsed = time.time() - t0

time.sleep(2)
after, _, per = mm_rows()
print(f"  elapsed {elapsed:.1f}s")
print(f"  AFTER  mm_rows={after}", flush=True)
print(f"  DELTA  mm_rows={after - before}")
if after == before:
    print("  VERDICT: mm autotuner did NOT fire -> mm is not on FlagGems -> whitelist ACTIVE")
else:
    print("  VERDICT: mm autotuner FIRED -> mm IS on FlagGems -> whitelist NOT active")
