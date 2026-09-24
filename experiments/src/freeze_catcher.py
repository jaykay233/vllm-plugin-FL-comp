#!/usr/bin/env python3
"""冻结捕获器：检测 engine 进程 CPU 骤降，抓取全部线程的 wchan/state 分布。

判定冻结：engine CPU < 60 jiffies/s 连续 3 个 0.5s 采样（正常 decode ~130+/s）。
抓取时对 /proc/<pid>/task/* 统计 (comm, state, wchan) 并按 wchan 聚合，
与「正常期」的分布做差，找出只在冻结期出现的等待点。
"""
import collections
import os
import subprocess
import sys
import time

PID = int(sys.argv[1])
OUT = sys.argv[2]
THRESH_SPS = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

JP = os.sysconf("SC_CLK_TCK")


def jiffies(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            p = f.read().rsplit(")", 1)[1].split()
        return int(p[11]) + int(p[12])
    except Exception:
        return None


def snapshot(pid):
    threads = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except Exception:
        return threads
    for tid in tids:
        base = f"/proc/{pid}/task/{tid}"
        try:
            comm = open(f"{base}/comm").read().strip()
        except Exception:
            comm = "?"
        try:
            st = open(f"{base}/stat").read().rsplit(")", 1)[1].split()[0]
        except Exception:
            st = "?"
        try:
            wchan = open(f"{base}/wchan").read().strip() or "-"
        except Exception:
            wchan = "?"
        threads.append((comm, st, wchan))
    return threads


def agg(threads):
    c = collections.Counter(wchan for _, _, wchan in threads)
    return c


with open(OUT, "w") as f:
    f.write("# 冻结窗口捕获  pid=%d  阈值=%.0f jiffies/s\n" % (PID, THRESH_SPS))
    prev = jiffies(PID)
    tprev = time.time()
    low = 0
    caught = 0
    # 正常期分布（用于对照）
    baseline_agg = None
    f.flush()

    while True:
        time.sleep(0.5)
        now = jiffies(PID)
        tnow = time.time()
        if now is None:
            f.write("[%s] engine 退出\n" % time.strftime("%H:%M:%S"))
            f.flush()
            sys.exit(0)
        dt = max(0.1, tnow - tprev)
        sps = (now - prev) / dt
        prev, tprev = now, tnow

        if sps < THRESH_SPS:
            low += 1
        else:
            low = 0
            # 记录一次正常期分布作对照
            if baseline_agg is None or time.time() % 30 < 1:
                baseline_agg = agg(snapshot(PID))

        if low == 3:
            caught += 1
            threads = snapshot(PID)
            a = agg(threads)
            f.write("\n" + "=" * 78 + "\n")
            f.write("[%s] 捕获冻结 #%d   CPU=%.0f jiffies/s  线程数=%d\n"
                    % (time.strftime("%H:%M:%S"), caught, sps, len(threads)))
            f.write("-" * 78 + "\n")
            f.write("当前 wchan 分布 (count, wchan):\n")
            for w, n in a.most_common():
                extra = ""
                if baseline_agg is not None:
                    d = n - baseline_agg.get(w, 0)
                    if d:
                        extra = f"   [对照差 {d:+d}]"
                f.write(f"   {n:4d}  {w}{extra}\n")
            # 冻结期较对照新增的等待点
            if baseline_agg is not None:
                new = {w: n - baseline_agg.get(w, 0) for w, n in a.items()
                       if n - baseline_agg.get(w, 0) > 0}
                f.write("仅在冻结期增多的 wchan:\n")
                if new:
                    for w, n in sorted(new.items(), key=lambda x: -x[1]):
                        f.write(f"   +{n:3d}  {w}\n")
                else:
                    f.write("   (无新增 -> 只是同一批线程睡得更久)\n")
            f.write("按进程名聚合:\n")
            for c, n in collections.Counter(t[0] for t in threads).most_common(8):
                f.write(f"   {n:4d}  {c}\n")
            f.flush()
        elif low > 3 and low % 20 == 0 and caught:
            # 持续冻结：继续记录，但稀疏一些
            f.write("[%s] 冻结持续中 CPU=%.0f jiffies/s\n"
                    % (time.strftime("%H:%M:%S"), sps))
            f.flush()
