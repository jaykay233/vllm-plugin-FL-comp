#!/usr/bin/env python3
"""把 vLLM 逐迭代日志按"轮次"切分，逐轮给出吞吐与长尾。

用法: iter_per_run.py <server.log> [gap_seconds]

benchmark_throughput_serve.py 在轮次之间有 cooldown，且每轮开始前客户端要
tokenize（约 50 s），因此相邻迭代的时间戳会出现明显空档。用空档切分就能把
每一轮分开，从而定位"低峰轮"是整体变慢还是被个别长尾步拖垮。

此外会把每轮的迭代耗时汇总与墙钟时长对比：
  迭代耗时合计 < 墙钟时长  => 引擎有空转/停顿，差额就是停顿时间
"""
import re
import sys
from datetime import datetime

LINE = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*?Iteration\((\d+)\): "
    r"(\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms"
)


def parse(path):
    out = []
    with open(path, errors="ignore") as f:
        for raw in f:
            m = LINE.search(raw)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%m-%d %H:%M:%S")
            out.append(
                dict(
                    ts=ts,
                    idx=int(m.group(2)),
                    ctx_req=int(m.group(3)),
                    ctx_tok=int(m.group(4)),
                    gen_req=int(m.group(5)),
                    gen_tok=int(m.group(6)),
                    ms=float(m.group(7)),
                )
            )
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/bench_results/iterlog/server.log"
    gap = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    iters = parse(path)
    if not iters:
        print(f"没有解析到 Iteration 行: {path}")
        return

    segs, cur = [], [iters[0]]
    for prev, nxt in zip(iters, iters[1:]):
        if (nxt["ts"] - prev["ts"]).total_seconds() > gap:
            segs.append(cur)
            cur = []
        cur.append(nxt)
    segs.append(cur)

    # 丢掉纯 warmup 段（只有 ctx、没有 gen，且迭代极少）
    real = [s for s in segs if any(i["gen_tok"] > 0 for i in s)]
    print(f"总迭代 {len(iters)}，切出 {len(segs)} 段，其中含 decode 的轮次 {len(real)} 个\n")

    print(f"{'#':>3} {'起':>8} {'迭代':>7} {'墙钟s':>8} {'迭代和s':>9} "
          f"{'空隙s':>7} {'输出tok/s':>10} {'in+out tok/s':>13} {'max ms':>8}")
    for k, s in enumerate(real, 1):
        wall = (s[-1]["ts"] - s[0]["ts"]).total_seconds() + s[-1]["ms"] / 1000
        busy = sum(i["ms"] for i in s) / 1000
        gen = sum(i["gen_tok"] for i in s)
        ctx = sum(i["ctx_tok"] for i in s)
        mx = max(i["ms"] for i in s)
        print(f"{k:>3} {s[0]['ts'].strftime('%H:%M:%S'):>8} {len(s):>7} "
              f"{wall:>8.1f} {busy:>9.1f} {wall-busy:>7.1f} "
              f"{gen/wall:>10.1f} {(gen+ctx)/wall:>13.1f} {mx:>8.1f}")

    print("\n逐轮长尾（相对该轮纯 decode 迭代中位数的 >3x 迭代）:")
    for k, s in enumerate(real, 1):
        dec = [i["ms"] for i in s if i["ctx_tok"] == 0]
        if not dec:
            continue
        med = sorted(dec)[len(dec) // 2]
        slow = [i for i in s if i["ms"] > 3 * med]
        extra = sum(i["ms"] - med for i in slow) / 1000
        tag = "  <-- 有空隙" if any(
            (b["ts"] - a["ts"]).total_seconds() > 1.0
            for a, b in zip(s, s[1:])
        ) else ""
        print(f"  轮{k}: decode中位 {med:6.2f} ms  长尾 {len(slow):>3} 个  "
              f"额外 {extra:6.2f} s  max {max(i['ms'] for i in s):7.1f} ms{tag}")

    # 全日志里最大的迭代间空档，用于定位真正的停顿
    # 注意：只能按时长排序，元组里含 dict 时并列会比较 dict 而报错
    gaps = sorted(
        (
            ((b["ts"] - a["ts"]).total_seconds(), a, b)
            for a, b in zip(iters, iters[1:])
            if (b["ts"] - a["ts"]).total_seconds() > 1.0
        ),
        key=lambda t: t[0],
    )
    print(f"\n迭代间 >1s 的空档共 {len(gaps)} 处，最长的 10 处:")
    for g, a, b in gaps[-10:]:
        print(f"  {g:>6.1f} s  iter{a['idx']}->iter{b['idx']}  "
              f"({a['ts'].strftime('%H:%M:%S')})  "
              f"前一步 ctx={a['ctx_tok']} gen={a['gen_tok']} {a['ms']:.1f}ms")

    # 空档的成因是"引擎停顿"还是"客户端还没发请求"，用这一步区分：
    # 空档前一步若是 context-only（gen_tok==0）且后一步立刻恢复 gen，
    # 说明请求侧断流；若前后都在 gen，则是引擎自己停了。
    print("\n空档成因归类:")
    by_cause = {"前一步无 gen（请求断流）": 0, "前后都在 gen（引擎停顿）": 0,
                "其他": 0}
    worst = {}
    for g, a, b in gaps:
        if a["gen_tok"] == 0 and b["gen_tok"] > 0:
            k = "前一步无 gen（请求断流）"
        elif a["gen_tok"] > 0 and b["gen_tok"] > 0:
            k = "前后都在 gen（引擎停顿）"
        else:
            k = "其他"
        by_cause[k] += 1
        worst[k] = max(worst.get(k, 0.0), g)
    for k, v in by_cause.items():
        if v:
            print(f"  {k}: {v} 处，最长 {worst[k]:.1f} s")


if __name__ == "__main__":
    main()
