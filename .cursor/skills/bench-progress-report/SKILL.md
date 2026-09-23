---
name: bench-progress-report
description: >-
  Run long benchmarks, sweeps, and wait jobs in the background while posting a
  one-line progress report every minute, plus an automatic hang alert when
  output stalls. Use when running a benchmark, parameter sweep, long eval, or
  any job expected to take more than a few minutes, or when the user asks to be
  kept posted, asks for progress, or asks whether a job is stuck.
disable-model-invocation: true
---

# Benchmark / Long-Job Progress Reporting

## When to apply

Apply when a job is expected to run for more than ~3 minutes:

- the user asks to run a benchmark, sweep, sweep a parameter, or a long eval
- the user says "keep me posted", "report progress", "每分钟汇报", "有结果告诉我"
- the user asks "卡住了么" / "is it stuck" about a running job

Skip entirely for commands that finish in seconds — a per-minute report would
just be noise.

Invoked as `/bench-progress-report`. To make it fire automatically whenever a
long job is started, remove the `disable-model-invocation` line above.

## Core rule: make the job observable before reporting

A blocking foreground call cannot be reported on mid-flight, so the first step
is not reporting — it is converting the job into a background job with a stable
log path. Do this at launch:

```bash
cd <project> && source /opt/conda/etc/profile.d/conda.sh && conda activate <env> && \
python bench.py --cases 0,1,4 > /tmp/bench_<tag>.log 2>&1; echo "exit=$?"
```

- launch with `block_until_ms: 0` so the shell returns immediately
- use a **stable, unique** log path (`/tmp/bench_<tag>.log`); it becomes the
  source of truth for the whole run
- note and remember: log path, pid, expected number of cases, the metric the
  user cares about, and any baseline to compare against

Then attach the once-per-minute notification to the launch call itself:

```
notify_on_output: { pattern: "<regex matching per-case result lines>", debounce_ms: 60000, reason: "benchmark progress" }
```

Keep the pattern tight — it should match only the lines that carry a result
(e.g. `^\s*(case|ns)=|tok/s|TPOT`), never all output.

## Reporting loop

Report after each 60-second slice:

- poll with `AwaitShell` (`block_until_ms: 60000`); pass the shell id when
  polling a specific background job
- **never** use shell `sleep` to wait — it blocks the turn and produces no
  updates
- after each slice, read only what is **new**, then post exactly one line

Cheap incremental read (avoids re-reading a growing log):

1. `Grep` the log for the result pattern with `output_mode: "count"` → total N
2. you already reported up to Nₖ; fetch only the tail with
   `Read(path=<log>, offset=-30)`, or `Grep` with `offset: Nₖ`
3. if there is nothing new, say so in one line — do not re-print old lines

## One-line report template

```
[<hh:mm> +<elapsed>] <done>/<total> · <latest result vs baseline> · ETA ~<n>m · <log path>
```

Write the report in the user's language; keep it to a single line.

```
[15:42 +3m12s] 2/5 · ns=16 → TPOT 7134us (−5.1% vs baseline) · ETA ~2m · /tmp/bench_ctx.log
[15:42 +3m12s] 2/5 cases · no new result this minute · /tmp/bench_ctx.log
```

## Hang alert

Treat the job as stalled when **both** hold for **2 consecutive slices**:

- the log gained no new bytes, and
- no new result line appeared

Then immediately, in one message:

1. report the pid and whether it is still alive (`ps -p <pid>`)
2. name the last stage reached, quoting the last non-empty log line
3. give the likely cause and the single command you would run to diagnose

Do not silently keep polling a stalled job, and do not restart or kill it
without asking.

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
- **VRAM teardown races.** On a shared GPU, back-to-back runs can fail with
  `Free memory on device (...) is less than desired GPU memory utilization`.
  Check free memory before starting the next run instead of losing a slice
  to the failure.
- Never kill a job the user started without asking.
- Do not poll a job whose tool result says it was "manually backgrounded by the
  user" — report on it when it completes instead.
- Do not repeat the accumulated results table every minute; it belongs in the
  final report.
