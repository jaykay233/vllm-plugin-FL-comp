#!/usr/bin/env python3
"""把 1s 的 GPU/CPU 采样与 10s 的 engine 观测窗口关联，判定 0 吞吐窗口的性质。

采样列: ts,epoch,gpu_util,apiserver_state,apiserver_jiff,apiserver_thr,engine_state,engine_jiff,engine_thr

对每个 engine 观测窗口（10s），取落在其中的采样点，算出：
  gpu_mean      窗口内平均 GPU 利用率
  cpu_rate      窗口内 engine 进程的 CPU 占用率（jiffies/s / 100）
  states        窗口内 engine 进程状态分布

再按窗口的 generation throughput 分类：
  FROZEN  decode 期内 gen==0 且 prompt==0（引擎完全冻结）
  BUSY    gen 高
然后对比 FROZEN 与 BUSY 窗口的 gpu_mean / cpu_rate：
  FROZEN 的 gpu≈0 且 cpu 不高于 BUSY  -> host 侧阻塞（在等某件事）
  FROZEN 的 gpu≈0 且 cpu 显著更高     -> host 侧忙等/重算
  FROZEN 的 gpu 高                   -> device 侧卡顿
"""
import re
import sys
from datetime import datetime

SAMPLES = "/root/bench_results/tail_stall/samples.csv"
SERVER = "/root/bench_results/tail_stall/server.log"

WIN = re.compile(
    r"(?P<ts>\d\d-\d\d \d\d:\d\d:\d\d).*?\[loggers\.py:273\] Engine \d+: "
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%"
)


def sec(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def load_samples():
    rows = []
    for ln in open(SAMPLES):
        p = ln.strip().split(",")
        if len(p) < 9 or p[0] == "ts":
            continue
        rows.append({
            "ts": p[0], "sec": sec(p[0]),
            "gpu": int(p[2]) if p[2].lstrip("-").isdigit() else -1,
            "engine_state": p[6],
            "engine_jiff": int(p[7]) if p[7].isdigit() else 0,
        })
    return rows


def load_windows():
    rows = []
    for raw in open(SERVER, errors="ignore"):
        for ln in raw.replace("\r", "\n").split("\n"):
            m = WIN.search(ln)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "ts": d["ts"][-8:], "sec": sec(d["ts"][-8:]),
                "prompt": float(d["prompt"]), "gen": float(d["gen"]),
                "running": int(d["running"]), "waiting": int(d["waiting"]),
                "kv": float(d["kv"]),
            })
    return rows


def stats(samples, t0, t1):
    sel = [s for s in samples if t0 <= s["sec"] <= t1]
    if not sel:
        return None
    gpus = [s["gpu"] for s in sel if s["gpu"] >= 0]
    j0, j1 = sel[0]["engine_jiff"], sel[-1]["engine_jiff"]
    dt = max(1, sel[-1]["sec"] - sel[0]["sec"])
    # jiffies 是 USER_HZ=100 的 tick 数
    cpu_pct = (j1 - j0) / 100.0 / dt * 100.0
    states = {}
    for s in sel:
        states[s["engine_state"]] = states.get(s["engine_state"], 0) + 1
    return {
        "n": len(sel),
        "gpu": sum(gpus) / len(gpus) if gpus else -1,
        "gpu_max": max(gpus) if gpus else -1,
        "cpu_pct": cpu_pct,
        "states": states,
    }


def main():
    samples = load_samples()
    windows = load_windows()
    print(f"采样 {len(samples)} 行, 观测窗口 {len(windows)} 个\n")

    # 只看 decode 期（prompt 流量小）且 Running>0 的窗口
    rows = []
    for i, w in enumerate(windows):
        if w["running"] == 0:
            continue
        t1 = w["sec"]
        t0 = t1 - 10
        st = stats(samples, t0, t1)
        if st is None:
            continue
        frozen = w["prompt"] < 1.0 and w["gen"] < 1.0
        rows.append({**w, **st, "frozen": frozen})

    dec = [r for r in rows if r["prompt"] < 100.0]
    fro = [r for r in dec if r["frozen"]]
    busy = [r for r in dec if not r["frozen"]]

    def agg(rs, label):
        if not rs:
            print(f"{label}: 无样本")
            return
        g = sum(r["gpu"] for r in rs) / len(rs)
        gm = sum(r["gpu_max"] for r in rs) / len(rs)
        c = sum(r["cpu_pct"] for r in rs) / len(rs)
        st = {}
        for r in rs:
            for k, v in r["states"].items():
                st[k] = st.get(k, 0) + v
        print(f"\n{label}  (n={len(rs)})")
        print(f"  GPU 利用率  均值={g:6.1f}%  峰值={gm:6.1f}%")
        print(f"  engine CPU  {c:6.1f}%   (jiffies 推算)")
        print(f"  engine 状态分布 {st}")

    print("=" * 76)
    print("decode 期窗口分类")
    print("=" * 76)
    agg(busy, "BUSY  正常生成窗口")
    agg(fro, "FROZEN 零吞吐窗口")

    if fro and busy:
        gb = sum(r["gpu"] for r in busy) / len(busy)
        gf = sum(r["gpu"] for r in fro) / len(fro)
        cb = sum(r["cpu_pct"] for r in busy) / len(busy)
        cf = sum(r["cpu_pct"] for r in fro) / len(fro)
        print("\n" + "=" * 76)
        print("判定")
        print("=" * 76)
        print(f"  GPU: FROZEN {gf:.1f}% vs BUSY {gb:.1f}%   (差 {gf-gb:+.1f} pp)")
        print(f"  CPU: FROZEN {cf:.1f}% vs BUSY {cb:.1f}%   (差 {cf-cb:+.1f} pp)")
        if gf < 5 and cf <= cb + 5:
            print("  => GPU 空闲且 CPU 不忙：host 侧【在等待】某件事")
            print("     (驱动同步 / 内存分配 / IPC / 图重放失败重试)")
        elif gf < 5 and cf > cb + 5:
            print("  => GPU 空闲但 CPU 忙：host 侧【忙等或重算】")
        elif gf > 20:
            print("  => 冻结期 GPU 仍忙：device 侧卡顿（长 kernel / 同步屏障）")
        else:
            print("  => 证据不足，需更细粒度采样")

    print("\n所有 FROZEN 窗口明细:")
    for r in sorted(fro, key=lambda x: x["gpu"]):
        print(f"  {r['ts']}  gpu={r['gpu']:5.1f}% (max {r['gpu_max']:3.0f})  "
              f"cpu={r['cpu_pct']:6.1f}%  run={r['running']:3d} wait={r['waiting']:3d} "
              f"kv={r['kv']:5.1f}%  states={r['states']}")


if __name__ == "__main__":
    main()
