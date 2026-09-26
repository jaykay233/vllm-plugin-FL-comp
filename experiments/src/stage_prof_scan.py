#!/usr/bin/env python3
"""汇总 VLLM_ITER_STAGE_PROFILE=1 埋点的输出（新版 8 段）。

用法: stage_prof_scan.py <server.log>

埋点行（由 experiments/patches/vllm_iter_stage_profile.patch 产生）:

  [stage-prof] win <win>s n=<n> | input <s> (<ms>/it) loop_step <s> (<ms>/it)
               || in step: sched <s> (<ms>) exe <s> (<ms>) upd <s> (<ms>)
               || detail: gram <s> (<ms>) sample <s> (<ms>) wait <s> (<ms>)
                          resid <s> (<ms>) idle <s> (<ms>) unmarked <s>
  [stage-prof] slow loop <总>s | step_total=<s> | this iter: input=.. sched=..
                exec=.. gram=.. sample=.. wait=.. update=.. resid=..

各段含义
  input   整个 _process_input_queue() 轮询（含阻塞等待）
  idle    input 里真正阻塞在 input_queue.get() 的部分（空闲时≈input，即休眠）
  sched   scheduler.schedule()
  exec    model_executor.execute_model() 提交（非阻塞）
  gram    scheduler.get_grammar_bitmask()
  sample  model_executor.sample_tokens() 提交（非阻塞）
  wait    future.result()，在 log_iteration_details 块内（这是 GPU 有效等待）
  upd     _process_aborts_queue() + scheduler.update_from_output()
  resid   loop_step 减去上面六段 == step_fn 自身开销 + 输出入队 + post_step
  unmarked= win - input - loop_step，闭合检查，必须≈0

关键读法（对应 FINDINGS §5.1 第 6 条）
  · 先看 unmarked：不为 ~0 就说明归因不可信，别急着解释谁最大。
  · wait 大 -> 在等 GPU；其余大 -> 在等 CPU，是可优化空间。
  · input 要先减掉 idle，剩下的 input_active 才是真实排空成本。
  · resid 是迭代内「没被任何标记覆盖」的时间，含输出入队与 post_step。
"""
import re
import sys

# 新版（8 段）。前缀里有 <时间戳> pid=<n>，用 .*? 容忍。
WIN_NEW = re.compile(
    r"stage-prof\].*?win ([\d.]+)s n=(\d+) \| input ([\d.]+)s \(([\d.]+)ms/it\) "
    r"loop_step ([\d.]+)s \(([\d.]+)ms/it\) \|\| in step: sched ([\d.]+)s "
    r"\(([\d.]+)\) exe ([\d.]+)s \(([\d.]+)\) upd ([\d.]+)s \(([\d.]+)\) "
    r"\|\| detail: exec ([\d.]+)s \(([\d.]+)\) gram ([\d.]+)s \(([\d.]+)\) "
    r"sample ([\d.]+)s \(([\d.]+)\) wait ([\d.]+)s \(([\d.]+)\) "
    r"resid ([\d.]+)s \(([\d.]+)\) idle ([\d.]+)s \(([\d.]+)\) "
    r"unmarked (-?[\d.]+)"
)
# 旧版（3 段，无 detail）
WIN_OLD = re.compile(
    r"stage-prof\].*?win ([\d.]+)s n=(\d+) \| input ([\d.]+)s \(([\d.]+)ms/it\) "
    r"loop_step ([\d.]+)s \(([\d.]+)ms/it\) \|\| in step: sched ([\d.]+)s "
    r"\(([\d.]+)\) exe ([\d.]+)s \(([\d.]+)\) upd ([\d.]+)s \(([\d.]+)\)"
)
SLOW_OLD = re.compile(
    r"stage-prof\].*?slow loop ([\d.]+)s \(input ([\d.]+)s, step ([\d.]+)s\)"
)
SLOW_NEW = re.compile(
    r"stage-prof\].*?slow loop ([\d.]+)s \| step_total=([\d.]+)s \| this iter: "
    r"input=([\d.]+)s sched=([\d.]+)s exec=([\d.]+)s gram=([\d.]+)s "
    r"sample=([\d.]+)s wait=([\d.]+)s update=([\d.]+)s resid=([\d.]+)s"
)

FIELDS = ("input", "idle", "sched", "exec", "gram", "sample", "wait", "upd",
          "resid", "loop_step")


def collect(path):
    wins, slow = [], []
    for line in open(path, errors="ignore"):
        m = WIN_NEW.search(line)
        if m:
            g = [float(x) for x in m.groups()]
            d = dict(zip(
                ("win", "n", "input", "input_ms", "loop_step", "loop_step_ms",
                 "sched", "sched_ms", "exe", "exe_ms", "upd", "upd_ms",
                 "exec", "exec_ms", "gram", "gram_ms", "sample", "sample_ms",
                 "wait", "wait_ms", "resid", "resid_ms", "idle", "idle_ms",
                 "unmarked"), (float(x) for x in g)))
            d["n"] = int(d["n"])
            d["has_detail"] = True
            wins.append(d)
            continue
        m = WIN_OLD.search(line)
        if m:
            g = [float(x) for x in m.groups()]
            # 旧版只有 exe 聚合；把 exe 全归给 exec，其余置 0
            d = dict(zip(
                ("win", "n", "input", "input_ms", "loop_step", "loop_step_ms",
                 "sched", "sched_ms", "exe", "exe_ms", "upd", "upd_ms"), g))
            d["n"] = int(d["n"])
            d.update({"gram": 0.0, "sample": 0.0, "wait": 0.0, "exec": d["exe"],
                      "resid": d["loop_step"] - d["sched"] - d["exe"] - d["upd"],
                      "idle": 0.0, "unmarked": float("nan"), "has_detail": False})
            wins.append(d)
            continue
        m = SLOW_NEW.search(line)
        if m:
            g = [float(x) for x in m.groups()]
            slow.append(dict(zip(
                ("total", "step_total", "input", "sched", "exec", "gram",
                 "sample", "wait", "upd", "resid"), g)))
            continue
        m = SLOW_OLD.search(line)
        if m:
            slow.append({"total": float(m.group(1)), "input": float(m.group(2)),
                         "step_total": float(m.group(3))})
    return wins, slow


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/root/bench_results/stage_prof/server.log"

    wins_all, slow = collect(path)
    if not wins_all:
        print(f"没有解析到 stage-prof 窗口行: {path}")
        print("排查顺序：")
        print("  1) 补丁是否还在  -> experiments/src/stage_prof_patch.sh status")
        print("  2) 是否设了开关   -> VLLM_ITER_STAGE_PROFILE=1")
        print("     且建议设 VLLM_ITER_STAGE_FILE=<独立文件>：EngineCore 的 stdout")
        print("     会被 vLLM 重定向/装饰，靠 stdout 的埋点曾两次什么都拿不到。")
        print("  3) 窗口是否够长   -> 默认 10s。也可只看 slow loop 行。")
        return 1

    # Steady-state filter: a window dominated by `idle` is the engine waiting
    # for requests (startup ramp / gaps between phases), not engine work.
    # Excluding it keeps the percentages meaningful, but we report the excluded
    # total so nothing is silently hidden.
    idle_wins = [w for w in wins_all if w["win"] > 0 and w["idle"] > 0.5 * w["win"]]
    wins = [w for w in wins_all if w not in idle_wins]
    if not wins:
        wins = wins_all
        idle_wins = []
    excluded = sum(w["win"] for w in idle_wins)

    tot = {k: sum(w.get(k, 0.0) for w in wins) for k in FIELDS}
    tot["win"] = sum(w["win"] for w in wins)
    n = sum(w["n"] for w in wins)
    win = tot["win"]
    unmarked = sum(w["unmarked"] for w in wins if w["unmarked"] == w["unmarked"])
    detailed = all(w["has_detail"] for w in wins)

    def ms(v):
        return v * 1000.0 / max(n, 1)

    def row(name, v, indent=0, mark=""):
        pad = "  " * indent
        print(f"{pad}{name:<30} {v:>9.3f} {v / win * 100:>6.1f}% "
              f"{ms(v):>8.2f}{mark}")

    print(f"窗口 {len(wins)} 个（稳态），覆盖墙钟 {win:.1f}s，引擎迭代 {n} 次")
    if idle_wins:
        print(f"另有 {len(idle_wins)} 个空闲主导窗口被排除，"
              f"合计 {excluded:.1f}s（idle>{50}% 视为等请求，非引擎工作）")
    print(f"（平均每次迭代 {win / max(n, 1) * 1000:.1f} ms，"
          f"迭代频率 {n / max(win, 1e-9):.1f} it/s）")
    if not detailed:
        print("\n[注意] 这些窗口来自旧版埋点（只有 3 段），gram/sample/wait/"
              "resid/idle 不可用。")
        print("       请先 stage_prof_patch.sh apply 再跑一轮。")
    print()
    print(f"{'阶段':<32} {'合计s':>9} {'占窗口':>7} {'ms/次':>8}")
    print("-" * 62)

    row("input 输入队列轮询（含空闲）", tot["input"])
    row("└─ idle 阻塞等请求（休眠）", tot["idle"], indent=1)
    row("└─ input_active 真实排空", tot["input"] - tot["idle"], indent=1)
    print()
    row("loop_step 一次 busyloop 迭代", tot["loop_step"])
    row("├─ sched  schedule()", tot["sched"], indent=1)
    row("├─ exec   execute_model 提交", tot["exec"], indent=1)
    row("├─ gram   get_grammar_bitmask", tot["gram"], indent=1)
    row("├─ sample sample_tokens 提交", tot["sample"], indent=1)
    row("├─ wait   future.result 阻塞", tot["wait"], indent=1, mark="  <- GPU 有效等待")
    row("├─ upd    aborts+update_from_output", tot["upd"], indent=1)
    row("└─ resid  入队+post_step+私活", tot["resid"], indent=1)
    print("-" * 62)

    # 闭合检查：win == input + loop_step
    closure = win - tot["input"] - tot["loop_step"]
    host = (tot["input"] - tot["idle"]) + tot["sched"] + tot["exec"] + \
        tot["gram"] + tot["sample"] + tot["upd"] + tot["resid"]
    print(f"{'闭合检查 win-input-loop_step':<32} {closure:>9.3f} "
          f"{closure / win * 100:>6.1f}%")
    if unmarked == unmarked:  # NaN check
        print(f"{'（埋点自报 unmarked 累计）':<32} {unmarked:>9.3f} "
              f"{unmarked / win * 100:>6.1f}%")
    print()

    print("=" * 62)
    print(f"GPU 有效等待 (wait)      : {tot['wait']:>8.3f}s  "
          f"{tot['wait'] / win * 100:>5.1f}%")
    print(f"host 侧开销合计          : {host:>8.3f}s  {host / win * 100:>5.1f}%")
    print(f"空闲/休眠 (idle)         : {tot['idle']:>8.3f}s  "
          f"{tot['idle'] / win * 100:>5.1f}%")
    print("=" * 62)

    if abs(closure) > max(0.05 * win, 1.0):
        print(f"\n!! 闭合检查未通过：unmarked={closure:.3f}s 偏大，"
              "归因不可信，先修埋点再解释数字（FINDINGS §5.1.6）。")

    # 最大 host 开销点名
    host_items = {
        "input_active": tot["input"] - tot["idle"],
        "sched": tot["sched"], "exec": tot["exec"], "gram": tot["gram"],
        "sample": tot["sample"], "upd": tot["upd"], "resid": tot["resid"],
    }
    top = sorted(host_items.items(), key=lambda kv: -kv[1])[:4]
    print("\nhost 侧开销排序（优化候选）:")
    for k, v in top:
        print(f"  {k:<14} {v:>8.3f}s  {v / win * 100:>5.1f}%  "
              f"{ms(v):>7.2f} ms/次")

    if any(w["wait"] > 0 for w in wins):
        denom = max(tot["wait"] + host, 1e-9)
        gpu_share = tot["wait"] / denom
        verdict = ("GPU 受限（wait 主导）" if gpu_share > 0.5
                   else "CPU/host 受限（host 开销主导）")
        print(f"\n判定: {verdict}")
        print(f"  在「busy 时间」(wait+host，不含 idle) 里，"
              f"等 GPU 占 {gpu_share:.1%}，host 开销占 {1 - gpu_share:.1%}")

    if slow:
        big = sorted(slow, key=lambda d: -d["total"])
        print(f"\n慢循环事件 {len(slow)} 个（阈值 VLLM_ITER_STAGE_SLOW）:")
        for d in big[:15]:
            print(f"  {d['total']:>8.3f}s  input={d.get('input', 0):.3f} "
                  f"step={d.get('step_total', 0):.3f} "
                  f"wait={d.get('wait', 0):.3f} exec={d.get('exec', 0):.3f} "
                  f"gram={d.get('gram', 0):.3f}")
        # Split by cause: a slow turn whose time sits in `input` is the engine
        # waiting for the client, NOT an engine stall. Only step-dominated slow
        # turns are engine problems. Summarise them separately, otherwise the
        # idle blocks (tens of seconds each) swamp the ratio.
        idle_slow = [d for d in slow if d.get("input", 0) > d.get("step_total", 0)]
        step_slow = [d for d in slow if d not in idle_slow]
        si = sum(d["total"] for d in idle_slow)
        ss = sum(d["total"] for d in step_slow)
        print(f"  ├─ input 主导（等客户端）: {len(idle_slow)} 个，合计 {si:.1f}s"
              f"  <- 非引擎问题")
        print(f"  └─ step 主导（引擎停顿）: {len(step_slow)} 个，合计 {ss:.1f}s"
              f"  占稳态窗口 {ss / win * 100:.1f}%")
        if ss > 0:
            print(f"     引擎停顿里 step_total 合计 "
                  f"{sum(d.get('step_total', 0) for d in step_slow):.1f}s，"
                  f"其中 wait(等GPU) {sum(d.get('wait', 0) for d in step_slow):.1f}s、"
                  f"exec {sum(d.get('exec', 0) for d in step_slow):.1f}s")
        print("  读法：input 大 = 在等客户端请求（非引擎问题）；"
              "step 大 + exec 大 = 引擎真停顿。")
    else:
        print("\n没有慢循环告警（未出现超过阈值的单次迭代）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
