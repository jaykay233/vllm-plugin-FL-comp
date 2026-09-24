---
name: bench-progress-report
description: >-
  Launch long benchmarks, sweeps, evals, and wait jobs in the background with an
  output notification hook attached at launch, so progress arrives without
  busy-waiting, plus an automatic hang alert when output stalls. Use when running
  a benchmark, parameter sweep, long eval, server startup, or any job expected to
  take more than a few minutes, or when the user asks to be kept posted, asks for
  progress, or asks whether a job is stuck.
disable-model-invocation: true
---

# Benchmark / Long-Job Progress Reporting

## When to apply

Apply when a job is expected to run for more than ~3 minutes:

- the user asks to run a benchmark, sweep a parameter, or a long eval
- the user says "keep me posted", "report progress", "每分钟汇报", "有结果告诉我"
- the user asks "卡住了么" / "is it stuck" about a running job
- you are about to run `benchmark_throughput_serve.py`, `evalscope eval`,
  `vllm serve` (cold start), or any other multi-minute shell command

Skip entirely for commands that finish in seconds — a hook would just be noise.

Invoked as `/bench-progress-report`.

## Core rule: hook at launch, never busy-wait

Two obligations, both **at launch time**, before the job is allowed to run:

1. **Background it** with `block_until_ms: 0` and a stable log path.
2. **Attach the hook** — `notify_on_output` on that same launch call.

A blocking foreground call cannot be reported on at all, and a job launched
without a hook can only be tracked by polling it. Polling is exactly the
busy-wait this skill exists to remove.

```bash
cd <project> && <command> > /tmp/bench_<tag>.log 2>&1; echo "exit=$?"
```

Launch that with:

```
block_until_ms: 0
notify_on_output: { pattern: "<result-line regex>", debounce_ms: 60000, reason: "benchmark progress" }
```

- keep the pattern tight: match only result-carrying lines, never all output
- `debounce_ms: 60000` for per-case results, `300000`+ for slow stages
- record the log path, pid, expected case count, the metric the user cares about,
  and the baseline to compare against

### Hook patterns known to work here

| job | pattern |
|---|---|
| `benchmark_throughput_serve.py` | `Total token throughput \(tok/s\):` |
| case-style sweep | `^\s*(case\|ns)=` |
| `evalscope eval` | `Accuracy ↑\|Overall report table` |
| `vllm serve` cold start | `Application startup complete\|Capturing CUDA graphs.*100%` |
| generic stage progress | `DONE\|done\|PASSED` |
| long sweep with stage lines | `^\[\d\d:\d\d:\d\d\]` |

Never hook on something that matches everything (`^`, `` , `\s`) — it fires on
every line and is worse than no hook.

## Waiting: one long block, not many short ones

After launching, wait with **one `AwaitShell` sized to the job**, not a
minute-by-minute poll loop:

- use `block_until_ms` of 600000–2700000 for jobs expected to run tens of minutes
- the hook delivers intermediate progress, so you do not need to poll for it
- when the block returns, read **only what is new**, then report or continue
- **never** use shell `sleep` to wait — it blocks the turn and produces nothing
- reach for a 60 s poll loop only when the user explicitly asked for a per-minute
  cadence, or the job can silently hang and needs active watching

Cheap incremental read, so a growing log is never re-read in full:

1. `Grep` the log for the hook pattern with `output_mode: "count"` → total N
2. you already reported up to Nₖ; fetch only the tail with
   `Read(path=<log>, offset=-30)`, or `Grep` with `offset: Nₖ`
3. if nothing is new, say so in one line — do not re-print old lines

## One-line report template

```
[<hh:mm> +<elapsed>] <done>/<total> · <latest result vs baseline> · ETA ~<n>m · <log path>
```

Write the report in the user's language; keep it to a single line.

```
[15:42 +3m12s] 2/5 · ns=16 → TPOT 7134us (−5.1% vs baseline) · ETA ~2m · /tmp/bench_ctx.log
[15:42 +3m12s] 2/5 cases · no new result this block · /tmp/bench_ctx.log
```

## Silent stage vs hang

No output is **not** the same as stuck. Before raising an alert, check what the
process is doing — the two long silent stages seen here are legitimate:

| symptom | legitimate explanation |
|---|---|
| log quiet for many minutes, CPU high | cold `torch.compile` / cudagraph capture (has been ~20 min here) |
| engine idle, benchmark client busy | client-side tokenization of `num_prompts × input_len` (has been ~50 s) |
| log quiet, process CPU ~0, state `S` | waiting on a slow dependency (model load, network FS) |

```bash
ps -o pid,stat,pcpu,etime -p <pid>      # alive and working, or sleeping on nothing?
wc -c < <log>                           # bytes actually gained
```

Only after that, treat the job as stalled when, across **2 consecutive blocks**:

- the log gained no new bytes, and
- no new result line appeared, and
- the process is alive but making no CPU progress

Then immediately, in one message:

1. report the pid and whether it is alive (`ps -p <pid>`)
2. name the last stage reached, quoting the last non-empty log line
3. give the likely cause and the single command you would run to diagnose

Do not silently keep waiting on a genuinely stalled job, and do not restart or
kill it without asking.

## Final report

When the job exits:

- one compact table of all cases (done/total, key metric, delta vs baseline)
- the winning configuration, stated explicitly
- the path to the raw log and any JSON/summary artifact

## Pitfalls

- **Concurrent writers.** Another sweep may be writing to the same log or
  results directory. Before trusting a number, check the file mtime and the
  producing pid — reading a file while it is being rewritten silently mixes
  runs and invalidates comparisons.
- **Shared GPU.** Check free memory and that no other session is mid-run before
  starting, or the run fails or silently competes. Back-to-back runs can also
  fail with `Free memory on device (...) is less than desired GPU memory
  utilization`; check before starting the next arm instead of losing a block.
- **Never kill a job you do not own.** Concurrent sessions share this machine;
  killing another session's server or benchmark is not yours to do.
- Do not poll a job whose tool result says it was "manually backgrounded by the
  user" — report on it when it completes instead.
- Do not repeat the accumulated results table each block; it belongs in the
  final report.
- **Python buffers stdout when it is not a tty.** A benchmark writing to a log
  can show an empty file while it is in fact running. Confirm progress from the
  server's own log or the process state before declaring a stall.
