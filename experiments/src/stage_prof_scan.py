#!/usr/bin/env python3
"""汇总 VLLM_ITER_STAGE_PROFILE=1 埋点的输出。

用法: stage_prof_scan.py <server.log>

埋点输出两类行：
  [stage-prof] win  <窗口> n=<迭代数> | input <s> (ms/it) loop_step <s> (ms/it)
               || in step: sched <s> (ms/it) exe <s> (ms/it) upd <s> (ms/it)
  [stage-prof] slow loop <总>s (input <s>, step <s>)

要点：`loop_step` 减去 step() 内部三项，差的都是 EngineCore 自己
（输出入队 + post_step）；而 input 是输入队列排空。这两块正是
`iteration elapsed time` 不覆盖的部分。
"""
import re
import sys

WIN = re.compile(
    r"stage-prof\] win ([\d.]+)s n=(\d+) \| input ([\d.]+)s \(([\d.]+)ms/it\) "
    r"loop_step ([\d.]+)s \(([\d.]+)ms/it\) \|\| in step: sched ([\d.]+)s "
    r"\(([\d.]+)\) exe ([\d.]+)s \(([\d.]+)\) upd ([\d.]+)s \(([\d.]+)\)"
)
SLOW = re.compile(
    r"stage-prof\] slow loop ([\d.]+)s \(input ([\d.]+)s, step ([\d.]+)s\)"
)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/root/bench_results/stage_prof/server.log"

    wins, slow = [], []
    for line in open(path, errors="ignore"):
        m = WIN.search(line)
        if m:
            g = m.groups()
            wins.append(dict(win=float(g[0]), n=int(g[1]), input=float(g[2]),
                             input_ms=float(g[3]), loop_step=float(g[4]),
                             loop_step_ms=float(g[5]), sched=float(g[6]),
                             sched_ms=float(g[7]), exe=float(g[8]),
                             exe_ms=float(g[9]), upd=float(g[10]),
                             upd_ms=float(g[11])))
            continue
        m = SLOW.search(line)
        if m:
            slow.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))

    if not wins:
        print(f"没有解析到 stage-prof 窗口行: {path}")
        print("（server 可能仍在启动/编译，或未设置 VLLM_ITER_STAGE_PROFILE=1）")
        return

    tot_n = sum(w["n"] for w in wins)
    tot_win = sum(w["win"] for w in wins)
    ti = sum(w["input"] for w in wins)
    tl = sum(w["loop_step"] for w in wins)
    ts = sum(w["sched"] for w in wins)
    te = sum(w["exe"] for w in wins)
    tu = sum(w["upd"] for w in wins)

    print(f"窗口 {len(wins)} 个，覆盖 {tot_win:.1f}s，迭代 {tot_n} 次\n")
    print(f"{'阶段':<34} {'合计s':>9} {'占比':>7} {'ms/次':>8}")
    print("-" * 62)

    def row(name, v):
        print(f"{name:<34} {v:>9.2f} {v/tot_win*100:>6.1f}% {v/tot_n*1000:>8.2f}")

    row("input 队列排空", ti)
    row("loop_step 合计", tl)
    print(f"  {'├─ sched (schedule+grammar)':<32} {ts:>9.2f} "
          f"{ts/tot_win*100:>6.1f}% {ts/tot_n*1000:>8.2f}")
    row("  ├─ exe (等模型 / sample)", te)
    row("  ├─ upd (aborts+update_from_output)", tu)
    row("  └─ 残余 (输出入队+post_step)", tl - ts - te - tu)
    print("-" * 62)
    row("未计时合计 (input + loop_step)", ti + tl)
    row("空隙 (窗口 - 未计时，即休眠)", tot_win - ti - tl)

    print("\n读法：")
    print("  · exe 是 GPU 执行等待，属于有效工作；其余都是 host 侧开销。")
    print("  · host 开销 = input + sched + upd + 残余，若其占比高，说明")
    print("    引擎在等 CPU 而不是等 GPU，属于可优化空间。")

    if slow:
        print(f"\n慢循环事件 {len(slow)} 个（阈值见 VLLM_ITER_STAGE_SLOW）:")
        big = sorted(slow, key=lambda t: -t[0])
        for dur, inp, stp in big[:20]:
            print(f"  {dur:>8.2f}s  (input {inp:.2f}s, step {stp:.2f}s)")
        print(f"  慢循环合计 {sum(s[0] for s in slow):.1f}s，"
              f"占覆盖时长 {sum(s[0] for s in slow)/tot_win*100:.1f}%")
    else:
        print("\n没有慢循环告警（本轮未出现 >阈值 的单次循环）")


if __name__ == "__main__":
    main()
