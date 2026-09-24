#!/usr/bin/env python3
"""扫描 vLLM 逐迭代日志，把"冻结"那次迭代的组成挖出来。

用法: iter_detail_scan.py <server.log>

日志格式（vllm/v1/engine/core.py log_iteration_details）:
  Iteration(N): X context requests, Y context tokens, Z generation requests,
                W generation tokens, iteration elapsed time: T ms

关注的不是平均值，而是长尾：一个 ~10 s 的迭代会把本轮吞吐整体拉低几个百分点，
正好对应实测到的 4927 / 5310 两个模式。
"""
import re
import statistics
import sys

LINE = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms"
)


def parse(path):
    iters = []
    with open(path, errors="ignore") as f:
        for raw in f:
            m = LINE.search(raw)
            if not m:
                continue
            iters.append(
                dict(
                    idx=int(m.group(1)),
                    ctx_req=int(m.group(2)),
                    ctx_tok=int(m.group(3)),
                    gen_req=int(m.group(4)),
                    gen_tok=int(m.group(5)),
                    ms=float(m.group(6)),
                )
            )
    return iters


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/bench_results/iterlog/server.log"
    iters = parse(path)
    if not iters:
        print(f"没有解析到 Iteration 行: {path}")
        print("（若 benchmark 仍在客户端 tokenize 阶段，服务端确实还没有迭代日志）")
        return

    ms = sorted(i["ms"] for i in iters)
    n = len(ms)
    med = statistics.median(ms)
    p99 = ms[min(n - 1, int(n * 0.99))]
    print(f"迭代总数 {n}")
    print(f"  median {med:.2f} ms   p90 {ms[int(n*0.90)]:.2f}   "
          f"p99 {p99:.2f}   max {ms[-1]:.2f}")
    print(f"  合计 {sum(ms)/1000:.1f} s")

    # 迭代按"是否含 prefill"分组，两者的正常耗时本来就不同量级
    dec = [i["ms"] for i in iters if i["ctx_tok"] == 0]
    pre = [i["ms"] for i in iters if i["ctx_tok"] > 0]
    if dec:
        print(f"\n纯 decode 迭代 {len(dec)}: median {statistics.median(dec):.2f} ms")
    if pre:
        print(f"含 prefill 迭代 {len(pre)}: median {statistics.median(pre):.2f} ms")

    # 冻结判据：显著超过同组中位数
    thr = 3.0 * med
    slow = sorted([i for i in iters if i["ms"] > thr], key=lambda x: -x["ms"])
    print(f"\n超过 3×median（{thr:.0f} ms）的迭代: {len(slow)} 个")
    if not slow:
        print("  → 本轮没有出现冻结长尾")
        return

    print(f"\n  {'idx':>6} {'ctx_req':>8} {'ctx_tok':>8} {'gen_req':>8} "
          f"{'gen_tok':>8} {'ms':>10}  {'vs_median':>10}")
    for i in slow[:25]:
        print(f"  {i['idx']:>6} {i['ctx_req']:>8} {i['ctx_tok']:>8} "
              f"{i['gen_req']:>8} {i['gen_tok']:>8} {i['ms']:>10.1f}  "
              f"{i['ms']/med:>9.1f}x")

    extra = sum(i["ms"] - med for i in slow) / 1000.0
    print(f"\n这些长尾迭代相对中位数的额外耗时合计: {extra:.1f} s "
          f"（占全部迭代耗时 {extra/(sum(ms)/1000)*100:.1f}%）")
    print("\n读法：若长尾迭代的 ctx_tok 很大 → 巨型 prefill chunk 独占引擎；")
    print("      若 ctx_tok=0 且 gen_tok 正常 → 纯 decode 步被外部事件阻塞；")
    print("      若 gen_tok 也异常大 → 是极宽的 decode batch。")


if __name__ == "__main__":
    main()
