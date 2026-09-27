#!/bin/bash
# 天数 BI-V150 官方口径评测：性能 (benchmark_throughput_serve) + 正确性 (evalscope math_500 Level 3)
#
# 与沐曦版 (run_eval_all.sh / run_eval_whitelist.sh) 的差异，逐条如下：
#
#   1. 运行时     沐曦用 /opt/conda/envs/mx（本镜像不存在），天数用系统
#                 /usr/local/bin/python3 + /usr/local/bin/evalscope。
#   2. serve 参数 天数官方命令行**带** --compilation-config
#                 '{"cudagraph_mode": "FULL_DECODE_ONLY"}'；沐曦**不带**。
#                 这是两者唯一的 serve 差异，也是必须保留的评测口径。
#   3. CUDAGraph   沐曦脚本 unset VLLM_FL_CUDAGRAPH_ONLY 是为了压掉 metax 的
#                 默认 CUDAGraph-only 路径。插件的
#                 _CUDAGRAPH_ONLY_DEFAULT_VENDORS 当前为空元组，天数没有这个
#                 默认值，因此这里既不 unset 也不 set，并在启动后断言它确实未设。
#   4. 基线/门槛   天数 total tok/s 基线 2028.01(4k) / 915.15(16k)，比沐曦低
#                 2.5x / 7.7x。**绝不能拿本机结果对比沐曦基线**，见
#                 .cursor/skills/race-s2-track2/SKILL.md。
#   5. 病态读数    天数 16k 的 TTFT 基线约 599 s，且 prefill/decode 反解会出现
#                 负的 decode 速率 —— 两阶段模型不描述这张卡。门槛照旧按
#                 ±1% 计算，但不要用阶段占比去推断优化收益。
#
# 用法:
#   bash /workspace/vllm-plugin-FL-comp/experiments/src/run_official_eval_iluvatar.sh
#   （建议经 launch_official_eval_iluvatar.sh 调用，避免边跑边改脚本）
#
# 环境变量可覆盖: EVAL_OUT / PORT / FRAMEWORK / MODEL / SERVED / SKIP_EVAL
set -u

# ------------------------------------------------------------------ 配置
PORT=${PORT:-9031}
MODEL=${MODEL:-/workspace/MiniCPM5-2B}
SERVED=${SERVED:-minicpm}
FRAMEWORK=${FRAMEWORK:-/workspace/vllm-plugin-FL}
DATASET=${DATASET:-/workspace/evalscope-datasets/math_500}
LEVEL_DIR=${LEVEL_DIR:-/workspace/evalscope-datasets/level3}
OUT=${EVAL_OUT:-/root/bench_results/eval_iluvatar}
CASES='[[4096,1024,64,256],[16384,1024,64,128]]'
CGRAPH_CFG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

PY=/usr/local/bin/python3
EVALSCOPE=/usr/local/bin/evalscope
BENCH=$FRAMEWORK/benchmarks/benchmark_throughput_serve.py

SERVER_LOG=$OUT/server.log
BENCH_LOG=$OUT/bench.log
EVAL_LOG=$OUT/evalscope.log

# 天数基线（组委会公布值）与 ±1% 硬门槛。来源: experiments/src/eval_baseline.py
#   case  baseline_total  baseline_ttft_ms
#   4k    2028.01         11573.53
#   16k    915.15         599262.03
GATE_4K_TOTAL_MIN=2007.73;  GATE_4K_TTFT_MAX=11689.27
GATE_16K_TOTAL_MIN=906.00;  GATE_16K_TTFT_MAX=605254.65
ACC_BASELINE=0.962;         ACC_MIN=0.95

ts()  { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

mkdir -p "$OUT"

# ---------------------------------------------------------- 预检 0：依赖版本
# 组委会要求：vllm-plugin-FL 分支 flagos-2026-s2 + FlagGems tag v5.3.5。
# 装错版本等于成绩无效，所以这里先断言，并且在最后再断言一次（跑久了可能被换）。
FJ_COMMIT=$(git -C "$FRAMEWORK" rev-parse --short HEAD 2>/dev/null || echo '?')
FJ_BRANCH=$(git -C "$FRAMEWORK" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')
CONFIGLOG=$OUT/preflight.txt
{
  echo "python      : $($PY -V 2>&1)"
  echo "framework   : $FRAMEWORK @ $FJ_BRANCH ($FJ_COMMIT)"
  echo "is in repo? : $(cd "$FRAMEWORK" 2>/dev/null && git log -1 --format=%s 2>/dev/null)"
  $PY - <<'PYEOF'
import importlib.metadata as m
for p in ("vllm", "vllm-plugin-fl", "flag_gems", "torch"):
    try:
        print(f"{p:16}: {m.version(p)}")
    except Exception as e:
        print(f"{p:16}: MISSING ({e})")
import flag_gems, vllm_fl
print("flag_gems live :", flag_gems.__file__)
print("vllm_fl  live :", vllm_fl.__file__)
PYEOF
} > "$CONFIGLOG" 2>&1
say "PART 0 | 预检"
cat "$CONFIGLOG"

# 天数官方题面要求 FlagGems 为 v5.3.5。本地可能是该 tag 的分支版（如
# 5.3.5.post1.devN+...）；只对**大版本偏离**告警，不硬拦，避免误伤。
if ! grep -aqE "flag_gems *: *5\.3\.5" "$CONFIGLOG"; then
  say "警告：flag_gems 版本不是 5.3.5 系列，与组委会要求(tag v5.3.5)不符。"
  say "      用 pip show flag_gems 确认；提交前必须换回 v5.3.5，否则成绩无效。"
fi

# ---------------------------------------------------------- 预检 1：单实例锁
# 踩过的坑：两个实例并存时，先启动那个的 server 崩了、就绪轮询仍在 grep 同一个
# server.log；后启动的 server 把 "Application startup complete" 写进同一文件，
# 于是僵死的旧实例被唤醒，两个 benchmark client 打同一个 server，结果互相污染。
LOCK=/tmp/run_official_eval_iluvatar.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  say "已有另一个评测实例在跑（$LOCK 被占用），本实例退出。"
  say "如需强制重跑，先确认旧实例已停：pgrep -af run_official_eval_iluvatar"
  exit 1
fi

# ------------------------------------------------- 预检 2：清理端口上的旧 server
# 归属判据（保守，宁漏勿误杀），只动下面两类：
#   1) 命令行含本 PORT 且含 vllm —— 本脚本自己的端口，属于上一轮我方 server；
#   2) ppid==1 的 VLLM::EngineCore 孤儿且 cwd 在 /workspace 下 —— 杀父进程不会
#      带走 EngineCore（它 re-parent 到 init），它会继续占着显存，导致新 server
#      报 `Free memory ... is less than desired` 而启动失败（已踩过一次）。
# 不匹配上面任何一条的进程一律不动，并打印出来供人工确认。
say "PART 0 | 清理 $PORT 端口上的旧 server 及其孤儿 EngineCore"
$PY - "$PORT" <<'PYEOF'
import os, signal, sys

port = sys.argv[1]
me = os.getpid()
killed, alive = [], []

for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        cl = open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore')
    except Exception:
        continue
    if not cl:
        continue
    try:
        cwd = os.readlink(f'/proc/{pid}/cwd')
    except Exception:
        cwd = '?'
    in_ws = cwd.startswith('/workspace')

    is_server = port in cl and 'vllm' in cl
    is_orphan_engine = cl.startswith('VLLM::EngineCore') and in_ws
    if is_server or is_orphan_engine:
        try:
            os.kill(int(pid), signal.SIGKILL)
            killed.append((pid, cwd, cl[:40]))
        except Exception:
            pass

for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cl = open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore')
    except Exception:
        continue
    if cl and ('VLLM::EngineCore' in cl or ('vllm' in cl and 'serve' in cl)):
        try:
            cwd = os.readlink(f'/proc/{pid}/cwd')
        except Exception:
            cwd = '?'
        alive.append((pid, cwd))

print("  killed:", [k[0] for k in killed] or "none")
for pid, cwd, cl in killed:
    print(f"    - {pid} cwd={cwd} {cl}")
print("  仍存活的 vLLM 进程（未动）:", alive or "none")
PYEOF
sleep 3

# ---------------------------------------------------------------- PART A: serve
say "PART A | 启动 vllm serve（天数评测口径：带 --compilation-config）"
cd "$FRAMEWORK" || exit 1

export VLLM_PLUGINS=fl
# 断言 VLLM_FL_CUDAGRAPH_ONLY 未设：它是沐曦口径的开关，天数基线不含它
if [ -n "${VLLM_FL_CUDAGRAPH_ONLY:-}" ]; then
  say "警告：VLLM_FL_CUDAGRAPH_ONLY=${VLLM_FL_CUDAGRAPH_ONLY} 已设置，会偏离天数评测口径，已 unset"
  unset VLLM_FL_CUDAGRAPH_ONLY
fi

# 就绪检测只认「本轮启动之后」写入的内容：先记录启动前的字节偏移。
: > "$SERVER_LOG"
OFFSET=0

nohup vllm serve "$MODEL" --port "$PORT" \
  --compilation-config "$CGRAPH_CFG" \
  --served-model-name "$SERVED" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  > "$SERVER_LOG" 2>&1 &
SRV_LAUNCHED=$!
SERVER_TREE="$SRV_LAUNCHED"
say "server pid=$SRV_LAUNCHED"

# 退出时收掉自己启动的整棵进程树。
# 踩过的坑：EngineCore 是子进程，杀掉父进程后它 re-parent 到 init 并继续占显存。
cleanup_server() {
  [ -z "${SERVER_TREE:-}" ] && return 0
  say "TEARDOWN | 停止本轮 server（含 EngineCore 子进程）"
  $PY - "$SERVER_TREE" <<'PYEOF'
import os, signal, sys, time
roots = [int(x) for x in sys.argv[1].split() if x.isdigit()]
pids = set(roots)
changed = True
while changed:
    changed = False
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            ppid = int(open(f'/proc/{pid}/stat').read().split()[3])
        except Exception:
            continue
        if ppid in pids and int(pid) not in pids:
            pids.add(int(pid)); changed = True
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
time.sleep(1)
left = [p for p in pids if os.path.isdir(f'/proc/{p}')]
print(f"  killed={sorted(pids)} still_alive={left or 'none'}")
PYEOF
}
trap cleanup_server EXIT

# 天数冷启动比沐曦慢：本镜像实测 5.4 版 FlagGems 约 90 s，v5.3.5 分支约 45 s；
# 给足 60 分钟以覆盖首次 triton 编译缓存冷的情况。
READY=0
for _ in $(seq 1 360); do
  if tail -c +$((OFFSET + 1)) "$SERVER_LOG" 2>/dev/null | grep -qa "Application startup complete"; then
    READY=1; break
  fi
  if ! kill -0 "$SRV_LAUNCHED" 2>/dev/null; then
    say "PART A | 失败：server 进程已退出。根因如下："
    tail -c +$((OFFSET + 1)) "$SERVER_LOG" | tr '\r' '\n' \
      | grep -aE "Error|error|ValueError|Traceback|RuntimeError|UnsupportedFlagTune" | tail -10
    say "提示：若报 triton.flagtune UnsupportedFlagTuneDeviceError，说明装了 5.4+ 的"
    say "      FlagGems（flagtune 探测不支持 corex backend）。换回 v5.3.5 tag，"
    say "      或临时 export USE_FLAGTUNE=0 绕过。"
    exit 1
  fi
  sleep 10
done
if [ "$READY" != 1 ]; then
  say "PART A | 失败，尾部日志："
  tail -c +$((OFFSET + 1)) "$SERVER_LOG" | tr '\r' '\n' | tail -15
  exit 1
fi
say "PART A | READY"

# 核实 --compilation-config 真的生效：CUDAGraph 捕获应当出现。
# 踩过的坑：脚本写对了参数不等于进程收到了参数，务必看 server 自己的输出。
tr '\r' '\n' < "$SERVER_LOG" | grep -aE "Loading weights took|GPU KV cache size|Maximum concurrency" | head -5
if tr '\r' '\n' < "$SERVER_LOG" | grep -qaE "Capturing CUDA graph|FULL_DECODE_ONLY|CUDAGraph"; then
  say "PART A | CUDAGraph 路径已在 server 日志中确认"
else
  say "PART A | 警告：日志未见 CUDAGraph 迹象，--compilation-config 可能未生效，本结果不可用于对照"
fi

curl -s -m 20 "http://localhost:$PORT/v1/models" | head -c 200; echo

# ---------------------------------------------------------------- PART B: 性能
say "PART B | 性能基准 (4k/16k, RUNS=4, SKIP_FIRST=1, 官方 2 个 case)"
cd /workspace || exit 1
# -u：重定向到文件时 Python 会缓冲 stdout，进程被信号带走会丢结果
$PY -u "$BENCH" \
  --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
  --test-cases "$CASES" \
  2>&1 | tee "$BENCH_LOG"
say "PART B | 完成"

# ------------------------------------------------------------ PART C: 正确性
if [ "${SKIP_EVAL:-0}" = "1" ]; then
  say "PART C | SKIP_EVAL=1，跳过正确性评测"
else
  say "PART C | 正确性评测 evalscope math_500 Level 3"
  rm -rf "$LEVEL_DIR"
  "$EVALSCOPE" eval \
    --model "$SERVED" \
    --api-url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets math_500 \
    --dataset-args "{\"math_500\": {\"dataset_id\": \"$DATASET\", \"subset_list\": [\"Level 3\"]}}" \
    --eval-batch-size 8 \
    --timeout 3600 \
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
    --work-dir "$LEVEL_DIR" \
    --ignore-errors \
    2>&1 | tee "$EVAL_LOG"
  say "PART C | 完成"
fi

# ------------------------------------------------------------ PART D: 判定
say "PART D | 对照天数门槛判定"
$PY - "$BENCH_LOG" "$EVAL_LOG" "$LEVEL_DIR" \
      "$GATE_4K_TOTAL_MIN" "$GATE_4K_TTFT_MAX" \
      "$GATE_16K_TOTAL_MIN" "$GATE_16K_TTFT_MAX" \
      "$ACC_BASELINE" "$ACC_MIN" <<'PYEOF'
import json
import os
import re
import sys

(bench_log, eval_log, level_dir,
 g4t, g4tt, g16t, g16tt, acc_base, acc_min) = sys.argv[1:10]
g4t, g4tt, g16t, g16tt = map(float, (g4t, g4tt, g16t, g16tt))
acc_base, acc_min = float(acc_base), float(acc_min)

# benchmark 脚本 summary 行:
#   Prefill=4096 Decode=1024 Conc=64 NumPrompts=256 Req/s=.. Total tok/s=.. TTFT=..ms
LINE = re.compile(
    r"Prefill=(\d+)\s+Decode=(\d+)\s+Conc=(\d+)\s+NumPrompts=(\d+)\s+"
    r"Req/s=([\d.]+)\s+Total tok/s=([\d.]+)\s+TTFT=([\d.]+)ms"
)
rows = {}
for line in open(bench_log, errors="ignore"):
    m = LINE.search(line)
    if m:
        pl, _, _, np_, _, tot, ttft = m.groups()
        rows[(int(pl), np_)] = (float(tot), float(ttft))

gates = {
    "4k": (g4t, g4tt),
    "16k": (g16t, g16tt),
}
key_by_case = {"4k": (4096, "256"), "16k": (16384, "128")}

print("=" * 78)
print(f"{'case':5} {'Total tok/s':>12} {'gate ≥':>10} {'verdict':>8}   "
      f"{'TTFT(ms)':>12} {'gate ≤':>12} {'verdict':>8}")
print("=" * 78)
ok_all = True
missing = []
for case, key in key_by_case.items():
    if key not in rows:
        missing.append(case)
        continue
    tot, ttft = rows[key]
    gt, gtt = gates[case]
    p_ok, t_ok = tot >= gt, ttft <= gtt
    ok_all &= p_ok and t_ok
    print(f"{case:5} {tot:12.2f} {gt:10.2f} {'PASS' if p_ok else 'FAIL':>8}   "
          f"{ttft:12.2f} {gtt:12.2f} {'PASS' if t_ok else 'FAIL':>8}")
print("=" * 78)
if missing:
    print(f"!! 未在 bench 日志中找到 case: {', '.join(missing)} —— 判定不完整")
    ok_all = False

# 正确性：优先读 json 报告，退化到 grep 日志
acc = None
cand = []
if os.path.isdir(level_dir):
    for root, _dirs, files in os.walk(level_dir):
        for f in files:
            if f == "math_500.json":
                cand.append(os.path.join(root, f))
for p in sorted(cand):
    try:
        data = json.load(open(p))
        for k in ("acc", "accuracy", "score", "AverageAccuracy"):
            if isinstance(data, dict) and k in data:
                acc = float(data[k]); break
        if acc is None and isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict):
                    for k in ("acc", "accuracy", "score"):
                        if k in v:
                            acc = float(v[k]); break
                if acc is not None:
                    break
    except Exception:
        pass
    if acc is not None:
        break

if acc is None:
    try:
        txt = open(eval_log, errors="ignore").read()
        m = re.findall(r"(?:AverageAccuracy|acc|accuracy)\D{0,12}([01]\.\d+)", txt, re.I)
        if m:
            acc = float(m[-1])
    except Exception:
        pass

print(f"accuracy baseline {acc_base}   gate ≥ {acc_min}")
if acc is None:
    print("accuracy: 未解析到（查看 .log / reports/ 下的 math_500.json）  INDETERMINATE")
    ok_all = False
else:
    print(f"accuracy: {acc:.4f}  {'PASS' if acc >= acc_min else 'FAIL'}")
    ok_all &= acc >= acc_min

print()
print("VERDICT:", "PASS - 达标" if ok_all else "FAIL / INDETERMINATE - 见上表")
print()
print("注：门槛是组委会的 ±1% 容差（tok/s ≥ baseline×0.99, TTFT ≤ baseline×1.01）。")
print("    **提升**要打败同会话基线，而不是仅仅越过门槛；")
print("    且本机为天数 BI-V150，不可与沐曦基线比较。")
PYEOF

echo
echo "================ 产物 ================"
echo "--- 各内部轮 ---"
grep -a "Total token throughput" "$BENCH_LOG" 2>/dev/null | tail -8
echo "--- summary ---"
grep -a "^Prefill=" "$BENCH_LOG" 2>/dev/null
echo "--- 正确性 ---"
grep -aiE "accuracy|Overall report table|math_500" "$EVAL_LOG" 2>/dev/null | tail -10
ls -la /workspace/benchmark_results/ 2>/dev/null | tail -4
ls -la "$LEVEL_DIR" 2>/dev/null | tail -6
echo
echo "preflight : $CONFIGLOG"
echo "server log: $SERVER_LOG"
echo "bench log : $BENCH_LOG"
echo "eval log  : $EVAL_LOG"
