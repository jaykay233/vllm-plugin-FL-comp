#!/bin/bash
set -u
OUT=/root/bench_results/eval_iluvatar_mathL3_fix_ext
LEVEL_DIR=/workspace/evalscope-datasets/level3_fix_ext
N=30
PORT=9048

for i in $(seq 1 720); do
  python3 - "$LEVEL_DIR" "$N" "$OUT" <<'PY'
import glob, json, os, sys
level_dir, n_target, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
paths = glob.glob(os.path.join(level_dir, "*", "reviews", "*", "*"))
rows = []
for p in paths:
    try:
        rows = [json.loads(line) for line in open(p) if line.strip()]
    except Exception:
        continue
    if rows:
        break
print(f"scored={len(rows)}", flush=True)
if len(rows) < n_target:
    sys.exit(1)

rows = sorted(rows, key=lambda r: r.get("index", 0))[:n_target]
wrong = []
correct = 0
for r in rows:
    try:
        acc = r["sample_score"]["score"]["value"]["accuracy"]
    except Exception:
        acc = 0.0
    if float(acc) >= 1.0:
        correct += 1
    else:
        wrong.append(r.get("index"))
acc = correct / n_target
summary = (
    f"partial Level3 n={n_target}\n"
    f"correct={correct}\n"
    f"accuracy={acc:.4f}\n"
    f"gate=0.95\n"
    f"verdict={'PASS_SMOKE (>=0.95)' if acc >= 0.95 else 'FAIL (<0.95)'}\n"
    f"wrong_indices={wrong}\n"
    f"config=SPLIT=1 once_per_step_meta FlagGems_UNUSED=ext graph_pages_fix, ATTN=1, MIN_Q=256, batch=8\n"
    f"baseline=noadd 29/30=0.9667\n"
    f"note=early-stop on first {n_target}/105; NOT official full Level3\n"
)
open(os.path.join(out, "partial30_summary.txt"), "w").write(summary)
print(summary, flush=True)
PY
  if [ $? -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] reached $N scored; stopping run"
    rg -ao "stats=\{[^}]*\}" "$OUT/server.log" | tail -1 | tee "$OUT/dispatch_stats.txt"
    pkill -f "evalscope eval" 2>/dev/null
    sleep 2
    pkill -f "run_level3.sh" 2>/dev/null
    sleep 2
    python3 - <<'PY'
import os, signal, time
killed = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except Exception:
        continue
    if "9048" in cmd or "EngineCore" in cmd or ("vllm" in cmd and "serve" in cmd):
        try:
            os.kill(int(pid), signal.SIGKILL)
            killed.append(pid)
        except Exception:
            pass
time.sleep(2)
print("killed", killed)
PY
    exit 0
  fi
  sleep 20
done
echo "[$(date +%H:%M:%S)] watcher timed out"
