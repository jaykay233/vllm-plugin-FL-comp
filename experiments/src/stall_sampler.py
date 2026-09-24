#!/usr/bin/env python3
"""每秒采样 GPU 利用率与关键进程 CPU/状态，用于判定 10s 冻结的性质。

输出列: ts,epoch,gpu_util,<label>_state,<label>_jiffies,<label>_threads,...

判读:
  gpu_util≈0 且 engine 状态 R 且 jiffies 高速增长 -> host 侧阻塞(CPU 忙)
  gpu_util≈0 且 engine 状态 S/D 且 jiffies 不动   -> 阻塞在外部(IO/IPC/驱动同步)
  gpu_util 高                                    -> device 侧卡顿
"""
import subprocess
import sys
import time

out = sys.argv[1]
procs = [a.split(":", 1) for a in sys.argv[2:]]  # label:pid


def stat_of(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        state = parts[2]
        jiff = int(parts[13]) + int(parts[14])
        threads = len(__import__("os").listdir(f"/proc/{pid}/task"))
        return state, jiff, threads
    except Exception:
        return "-", 0, 0


def gpu_util():
    try:
        r = subprocess.run(["mx-smi", "--show-usage"], capture_output=True,
                           text=True, timeout=5)
        for ln in r.stdout.splitlines():
            if ln.strip().startswith("GPU") and ":" in ln:
                return int(ln.split(":")[1].strip().rstrip("%").strip() or -1)
    except Exception:
        pass
    return -1


hdr = ["ts", "epoch", "gpu_util"]
for label, _ in procs:
    hdr += [f"{label}_state", f"{label}_jiff", f"{label}_thr"]
print(",".join(hdr), flush=True)

with open(out, "w") as f:
    f.write(",".join(hdr) + "\n")
    while True:
        t0 = time.time()
        row = [time.strftime("%H:%M:%S"), str(int(t0)), str(gpu_util())]
        for label, pid in procs:
            s, j, th = stat_of(pid)
            row += [s, str(j), str(th)]
        line = ",".join(row)
        f.write(line + "\n")
        f.flush()
        time.sleep(max(0.05, 1.0 - (time.time() - t0)))
