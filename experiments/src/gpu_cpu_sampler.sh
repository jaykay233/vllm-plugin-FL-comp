#!/bin/bash
# 每秒采样：GPU 利用率 + engine 进程 CPU/线程状态
# 用途：判定 10s 冻结窗口里 GPU 是空闲(host 阻塞) 还是忙碌(device 卡顿)
OUT=${1:-/root/bench_results/tail_stall/samples.csv}
SERVE_PID=${2:-}

echo "ts,epoch,gpu_util,eng_pid,eng_state,eng_cpu_jiffies,eng_threads" > "$OUT"

find_engine() {
  # EngineCore 是 serve 的子进程
  if [ -n "$SERVE_PID" ]; then
    pgrep -P "$SERVE_PID" 2>/dev/null | head -1
    return
  fi
  pgrep -f "EngineCore" 2>/dev/null | head -1
}

ENG=""
while true; do
  now=$(date +%s)
  ts=$(date +%H:%M:%S)
  util=$(mx-smi --show-usage 2>/dev/null | grep -aE "^\s+GPU\s+:" | grep -oE "[0-9]+ %" | head -1 | tr -d ' %')
  [ -z "$ENG" ] && ENG=$(find_engine)
  if [ -n "$ENG" ] && [ -d "/proc/$ENG" ]; then
    read -r state utime stime < <(awk '{print $3, $14, $15}' "/proc/$ENG/stat" 2>/dev/null)
    cpu=$(( ${utime:-0} + ${stime:-0} ))
    thr=$(ls /proc/$ENG/task 2>/dev/null | wc -l)
  else
    state="-"; cpu=0; thr=0
  fi
  echo "$ts,$now,${util:--1},${ENG:-0},${state:--},$cpu,$thr" >> "$OUT"
  sleep 1
done
