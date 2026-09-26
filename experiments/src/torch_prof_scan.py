#!/usr/bin/env python3
"""Summarise vLLM's torch-profiler table (profiler_out_<rank>.txt).

vLLM writes ONE table per rank via `--profiler-config.profiler=torch` with
`torch_profiler_dump_cuda_time_total=true`.

Two things worth knowing before you trust the numbers:

1. The table has BOTH CPU and CUDA columns, so a single file gives host-side and
   device-side attribution at the same time. vLLM only writes the CPU-only table
   when `activities` has exactly one entry, and it does not here
   (`activities=["CPU","CUDA"]` -> dump_cpu_time_total is False), so the file is
   not overwritten by a second table.

2. torch's own header labels are off by one for the device columns: the column
   labelled "Self CUDA" holds the self device TIME, and the column labelled
   "Self CUDA %" holds the TOTAL device percent. So this script never trusts the
   headers -- it indexes the numeric cells from the END of each row, which is
   stable regardless of how many leading text columns exist:

     name [overload]  <self cpu %> <self cpu> <cpu tot %> <cpu tot> <cpu avg>
                      <self cuda> <dev tot %> <cuda tot> <cuda avg>  <n calls>

   i.e. cells[-1]=calls, cells[-4]=self cuda, cells[-8]=self cpu.

Usage:
  torch_prof_scan.py <profiler_out_0.txt> [--top=25]
  torch_prof_scan.py <dir>            # finds profiler_out_*.txt
"""
import os
import re
import sys
from collections import defaultdict

CELL = re.compile(r"^-?[\d.]+(?:[eE][-+]?\d+)?(?:m|u|n)?s$|^-?[\d.]+%$")

TOTAL_CPU = re.compile(r"^Self CPU time total:\s*(\S+)$")
TOTAL_DEV = re.compile(r"^Self (\w+) time total:\s*(\S+)$")

# Coarse buckets so hundreds of kernels collapse into something actionable.
# First match wins.
BUCKETS = [
    ("attention (flash/paged)", r"flash|attn|attention|paged|prefill|decode_attn"),
    ("matmul/gemv", r"gemm|gemv|matmul|sgemm|hgemm|bgemm|cutlass|blas|"
                    r"triton_|batched_mm|linear|mm\b"),
    ("normalisation", r"rms|layernorm|layer_norm|norm|softmax"),
    ("activation/gating", r"silu|gelu|swiglu|mul_and|and_mul|act_|sigmoid|tanh"),
    ("rope/positional", r"rope|rotary|position|freqs"),
    ("copy/layout", r"copy|cat|contiguous|transpose|reshape|view|pad|slice|"
                    r"fill_|index|gather|scatter|select|stack|split|expand|"
                    r"repeat|clone|_to_copy|resize"),
    ("elementwise/arith", r"add|sub|mul|div|pow|where|clamp|fmin|fmax|compare|"
                          r"eq_|ne_|neg"),
    ("cast/quant", r"cast|convert|quant|dequant|bf16|fp16|fp8|half|float"),
    ("reduction/scan", r"reduce|sum|mean|argmax|argmin|topk|sort|cumsum|max\b|min\b"),
    ("sampling/logits", r"sample|logprob|penalt|multinomial|top_p|top_k|"
                        r"softmax|argmax"),
    ("allocator/sync", r"cudaMalloc|malloc|free|Memset|memcpy|synchronize|"
                       r"empty|zeros|TensorImpl|allocator"),
]


def to_seconds(s):
    m = re.match(r"^(-?[\d.]+(?:[eE][-+]?\d+)?)(m|u|n)?s$", s)
    if not m:
        return 0.0
    return float(m.group(1)) * {"m": 1e-3, "u": 1e-6, "n": 1e-9, None: 1.0}[m.group(2)]


def bucket_of(name):
    low = name.lower()
    for label, pat in BUCKETS:
        if re.search(pat, low):
            return label
    return "other"


def parse(path):
    rows, totals = [], {}
    for line in open(path, errors="ignore"):
        line = line.rstrip("\n")
        s = line.strip()
        m = TOTAL_CPU.match(s)
        if m:
            totals["cpu"] = to_seconds(m.group(1))
            continue
        m = TOTAL_DEV.match(s)
        if m:
            totals[m.group(1).lower()] = to_seconds(m.group(2))
            continue
        if not s or set(s) <= set("- "):
            continue

        toks = s.split()
        # The trailing "# of Calls" column is a bare integer with no unit, so it
        # must be peeled off before collecting the unit-bearing cells from the
        # end. Without this the scan stops on the very first token and finds
        # nothing at all.
        count = 0
        if toks and re.match(r"^\d+$", toks[-1]):
            count = int(toks[-1])
            toks = toks[:-1]

        cells, i = [], len(toks) - 1
        while i >= 0 and CELL.match(toks[i]):
            cells.insert(0, toks[i])
            i -= 1
        # 5 CPU cells + 4 device cells
        if len(cells) < 9:          # not a data row (header, note, stack line)
            continue
        name = " ".join(toks[: i + 1]).strip()
        if not name or name.lower().startswith("name"):
            continue
        cells.append(str(count))    # restore -> cells[-1] is the call count

        # Slice by BLOCK, not by raw offset: the two blocks have different
        # internal ordering, and getting that wrong silently yields zeros.
        #   CPU block: [self cpu %, self cpu, cpu total %, cpu total, cpu avg]
        #   dev block: [self dev, dev %, dev total, dev avg]   <- time FIRST
        # (torch's own header labels are also shifted for the device block; do
        #  not read this from the header text, see module docstring.)
        cpu_block = cells[-10:-5]
        dev_block = cells[-5:-1]
        rows.append({
            "name": name,
            "self_cpu": to_seconds(cpu_block[1]),
            "cpu_total": to_seconds(cpu_block[3]),
            "self_cuda": to_seconds(dev_block[0]),
            "cuda_total": to_seconds(dev_block[2]),
            "calls": count,
        })
    return rows, totals


def report(rows, key, title, top, unit):
    vals = [r for r in rows if r[key] > 0]
    if not vals:
        print(f"\n{title}: (无数据)")
        return 0.0
    tot = sum(r[key] for r in vals)
    print(f"\n{title}")
    print(f"  {len(vals)} 项，self 合计 {tot:.3f}s")
    print(f"  {'Name':<48} {'self':>10} {'share':>7} {'calls':>7}")
    print("  " + "-" * 76)
    for r in sorted(vals, key=lambda r: -r[key])[:top]:
        print(f"  {r['name'][:48]:<48} {r[key]:>10.4f} {r[key] / tot * 100:>6.1f}% "
              f"{r['calls']:>7}")

    by = defaultdict(float)
    byn = defaultdict(int)
    for r in vals:
        b = bucket_of(r["name"])
        by[b] += r[key]
        byn[b] += r["calls"]
    print(f"\n  按类别聚合:")
    print(f"  {'类别':<28} {'self s':>10} {'share':>7} {'calls':>8}")
    print("  " + "-" * 56)
    for label, sec in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<28} {sec:>10.4f} {sec / tot * 100:>6.1f}% "
              f"{byn[label]:>8}")
    return tot


def main():
    top = 22
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--top"):
            top = int(a.split("=", 1)[1]) if "=" in a else 22
        else:
            args.append(a)
    target = args[0] if args else "/root/bench_results/torch_prof_4k/prof"

    path = target
    if os.path.isdir(target):
        cands = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if f.startswith("profiler_out_") and f.endswith(".txt")]
        if not cands:
            print(f"目录里没有 profiler_out_*.txt: {target}")
            print("（profiler 未正常 stop，或 torch_profiler_dir 写错）")
            return 1
        path = cands[0]
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return 1

    rows, totals = parse(path)
    print(f"来源: {path}  ({os.path.getsize(path)} bytes)")
    if not rows:
        print("没解析到任何数据行。前 40 行原始内容：")
        for i, line in enumerate(open(path, errors="ignore")):
            if i >= 40:
                break
            print("  " + line.rstrip()[:160])
        return 1

    ctot = report(rows, "self_cuda", "GPU：按 self CUDA 时间（哪个 kernel 占 GPU）",
                  top, "s")
    htot = report(rows, "self_cpu", "HOST：按 self CPU 时间（哪个 op 占 CPU）",
                  top, "s")

    print("\n" + "=" * 62)
    print("总量与配比")
    if totals:
        for k, v in totals.items():
            print(f"  Self {k.upper():<5} time total (全事件): {v:.3f}s")
        if "cpu" in totals and "cuda" in totals and totals["cuda"] > 0:
            ratio = totals["cpu"] / totals["cuda"]
            print(f"  CPU/CUDA = {ratio:.2f}x"
                  f"  -> {'host 开销远超 GPU' if ratio > 1 else 'GPU 主导'}")
    print(f"  表内 self CUDA 合计 {ctot:.3f}s / self CPU 合计 {htot:.3f}s")
    print("=" * 62)
    print("\n配合 stage_prof_scan 一起读：")
    print("  stage 埋点答「哪个阶段占时间」；本表答「那个阶段花在哪个 op/kernel」。")
    print("  注意：本表是 profiler 开启窗口内的统计，steal 时间/被抢占不会体现，")
    print("        且 profiler 本身有开销，绝对值别和端到端吞吐直接换算。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
