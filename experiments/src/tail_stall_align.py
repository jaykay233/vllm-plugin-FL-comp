#!/usr/bin/env python3
"""按轮次对齐 vLLM engine 观测日志，定位 decode 期的尾部长停顿。

engine 每 ~10s 打印一行 loggers.py:273:
  Engine 000: Avg prompt throughput: X tok/s, Avg generation throughput: Y tok/s,
              Running: R reqs, Waiting: W reqs, GPU KV cache usage: K%,
              Prefix cache hit rate: P%

把连续活跃行聚成「轮」(round)，再在每轮里看 generation throughput 的
分布：健康 decode 期应当稳定在 ~1300-1400 tok/s；长停顿会让某些窗口整体掉下来。
"""
import re
import sys
from datetime import datetime

LINE = re.compile(
    r"(?P<ts>\d\d-\d\d \d\d:\d\d:\d\d).*?\[loggers\.py:273\] Engine \d+: "
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%, "
    r"Prefix cache hit rate: (?P<prefix>[\d.]+)%"
)


def parse(path):
    rows = []
    for raw in open(path, errors="ignore"):
        for ln in raw.replace("\r", "\n").split("\n"):
            m = LINE.search(ln)
            if not m:
                continue
            d = m.groupdict()
            t = datetime.strptime(d["ts"], "%m-%d %H:%M:%S")
            rows.append({
                "t": t,
                "sec": t.hour * 3600 + t.minute * 60 + t.second,
                "prompt": float(d["prompt"]),
                "gen": float(d["gen"]),
                "running": int(d["running"]),
                "waiting": int(d["waiting"]),
                "kv": float(d["kv"]),
                "prefix": float(d["prefix"]),
            })
    return rows


def segment(rows, gap=60.0):
    """聚成轮：连续的 Running>0 行，间隔 > gap 秒则切分。"""
    blocks, cur = [], []
    prev = None
    for r in rows:
        active = r["running"] > 0
        if not active:
            if cur:
                blocks.append(cur)
                cur = []
            prev = r
            continue
        if cur and prev is not None and (r["sec"] - prev["sec"]) > gap:
            blocks.append(cur)
            cur = []
        cur.append(r)
        prev = r
    if cur:
        blocks.append(cur)
    return [b for b in blocks if len(b) >= 3]


def dt_of(block, i):
    """第 i 行覆盖的时间区间（用相邻行时间差近似）。"""
    if i + 1 < len(block):
        return max(1.0, block[i + 1]["sec"] - block[i]["sec"])
    return max(1.0, block[i]["sec"] - block[i - 1]["sec"]) if len(block) > 1 else 10.0


def analyse(block, label):
    # 逐窗口积分出实际吞吐
    win = []
    for i, r in enumerate(block):
        win.append({**r, "dt": dt_of(block, i)})
    prompt_tok = sum(w["prompt"] * w["dt"] for w in win)
    gen_tok = sum(w["gen"] * w["dt"] for w in win)

    # decode 期窗口：期间几乎没有 prefill 流量
    dec = [w for w in win if w["prompt"] < 100.0]
    gens = sorted(w["gen"] for w in dec)
    med = gens[len(gens) // 2] if gens else 0.0
    stalls = [w for w in dec if w["gen"] < 0.5 * med] if med else []

    print(f"\n--- {label} ---")
    print(f"  时间 {block[0]['t']:%H:%M:%S} -> {block[-1]['t']:%H:%M:%S}  "
          f"({block[-1]['sec']-block[0]['sec']}s, {len(win)} 窗口)")
    print(f"  积分: prompt≈{prompt_tok/1e3:8.1f}k tok   gen≈{gen_tok/1e3:7.1f}k tok"
          f"   (CSV 期望 gen=262.1k)")
    if dec:
        print(f"  decode 窗口 {len(dec)} 个, gen tok/s: "
              f"min={gens[0]:7.1f} p10={gens[len(gens)//10]:7.1f} "
              f"med={med:7.1f} max={gens[-1]:7.1f}")
        print(f"  深度停顿(<0.5×med)窗口数 = {len(stalls)}"
              f"  占 decode 窗口 {100*len(stalls)/len(dec):.0f}%")
        tot_stall = sum((med - w['gen']) * w['dt'] for w in stalls)
        print(f"  停顿造成的损失 ≈ {tot_stall/1e3:.1f}k gen tok"
              f"  ({100*tot_stall/max(1,gen_tok):.1f}% of round)")
        for w in sorted(stalls, key=lambda x: x["gen"])[:6]:
            print(f"      {w['t']:%H:%M:%S}  gen={w['gen']:7.1f}  "
                  f"prompt={w['prompt']:7.1f}  Running={w['running']:3d}  "
                  f"Waiting={w['waiting']:3d}  KV={w['kv']:5.1f}%  dt={w['dt']:.0f}s")
    return {"gen": gen_tok, "stalls": len(stalls), "dec": len(dec)}


def main():
    for path, want in (
        ("/root/bench_results/eval_official/server.log", "我的 (adaptive)"),
        ("/root/bench_results/numsplit_ab/server_heuristic.log", "它的 (heuristic)"),
    ):
        print("=" * 84)
        print(f"# {want}   {path}")
        print("=" * 84)
        rows = parse(path)
        blocks = segment(rows)
        print(f"检测到 {len(blocks)} 轮")
        for idx, b in enumerate(blocks, 1):
            analyse(b, f"轮 {idx}")


if __name__ == "__main__":
    main()
