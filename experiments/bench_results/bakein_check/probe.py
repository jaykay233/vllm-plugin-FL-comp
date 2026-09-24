"""Decisive probe: is the platform-config flagos_whitelist actually active?

Under a working whitelist, `mm` is NOT dispatched to FlagGems, so FlagGems'
runtime Triton autotuner for mm never runs and the autotune DB never gains
`mm_kernel*` tables. Under a broken whitelist, brand-new prefill M values force
autotune and new mm tables appear.

We send prompts whose token counts are deliberately odd (1777, 1111) so the
prefill M values are very unlikely to already be cached.
"""

import json
import sqlite3
import time
import urllib.request

DB = "/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db"
URL = "http://127.0.0.1:9031/v1/completions"


def mm_tables():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    c.close()
    return len(tabs), len([t for t in tabs if t.startswith("mm_kernel")])


def send(prompt, ntok=4):
    body = json.dumps(
        {
            "model": "minicpm",
            "prompt": prompt,
            "max_tokens": ntok,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


# ~1 token per word; use odd word counts to get odd M values
base = "the quick brown fox jumps over a lazy dog and then "
n_total, n_mm = mm_tables()
print(f"  BEFORE  total_tables={n_total}  mm_tables={n_mm}", flush=True)

t0 = time.time()
for words in (1777, 1111, 1471):
    prompt = (base * (words // 10 + 1))[: words * 6]
    try:
        out = send(prompt)
        txt = out["choices"][0]["text"] if "choices" in out else out
        usage = out.get("usage", {})
        print(
            f"  sent words={words} prompt_tokens={usage.get('prompt_tokens')} "
            f"-> ok ({txt!r})",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  sent words={words} -> FAILED: {exc}", flush=True)

print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
time.sleep(2)
n_total, n_mm = mm_tables()
print(f"  AFTER   total_tables={n_total}  mm_tables={n_mm}", flush=True)
print(f"  VERDICT mm_tables_grew={n_mm > 0 and n_mm != 0}")
