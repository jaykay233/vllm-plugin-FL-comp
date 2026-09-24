---
name: eval-gated-optimization
description: >-
  Run the competition optimization loop for vllm-plugin-fl on MetaX: locate an
  optimization from profiling or test results, verify it with the official
  throughput/TTFT benchmark and the math_500 correctness eval, and auto-commit
  only when all four performance metrics improve. Use when optimizing
  throughput or TTFT for this repo, when asked to try an optimization and see if
  it helps, or when deciding whether a measured change is worth committing.
---

# Eval-Gated Optimization Loop

## When to apply

Apply when the task is "make the competition numbers better":

- trying a new kernel, dispatch path, attention config, or startup default
- deciding whether a measured change is worth keeping
- the user says "测一下这个优化", "有没有提升", "可以合入么", "继续优化"

Not for pure investigation with no candidate change — use
`self-directed-optimization` for that.

## The loop

```
1. Locate   from profiling / benchmark output, name the bottleneck and the
            candidate change. State the metric and the estimated gain.
2. Verify   run the official benchmark (4k + 16k) AND the correctness
            verification (accuracy always, plus the equivalence level that
            matches the change), against the SAME-SESSION baseline.
3. Gate     all 4 performance metrics improved AND correctness passed?
              yes -> commit the change
              no  -> revert / do not commit, go to 1 with what you learned
```

Never skip step 2 because the change "obviously helps", and never treat
correctness as optional — a correctness regression blocks the commit even when
all four performance metrics improve. Never commit from step 3 on a single run
(see Measurement rules).

## The 4 performance metrics

Two official cases, two metrics each — all four must move the right way:

| case | input | output | conc | num_prompts | Total tok/s gate | TTFT gate |
|---|---|---|---|---|---|---|
| 4k  | 4096  | 1024 | 64 | 256 | ≥ 5038.75 (baseline 5089.65) | ≤ 3231.43 ms (baseline 3199.44) |
| 16k | 16384 | 1024 | 64 | 128 | ≥ 6959.38 (baseline 7029.68) | ≤ 27469.11 ms (baseline 27197.14) |

The gates are the organiser's −1% / +1% tolerance. **Improvement means beating
the same-session baseline**, not merely clearing the gate: a change that clears
the gate but is a regression versus the pre-change run must not be committed.

Correctness is a **hard blocker**, not a tiebreaker: a correctness regression
vetoes the commit even when all four performance metrics improve. It is verified
by the three levels below — accuracy always, plus the equivalence check that
matches the kind of change.

## Commands

Launch the stock-config server first (the organiser's 沐曦 command — no
`--compilation-config`), because that is what the numbers are compared against:

```bash
export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072
```

Performance, from `/workspace` (the script writes `benchmark_results/` into the
CWD):

```bash
cd /workspace && python /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

Correctness:

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 --ignore-errors
```

### Every one of these is a long job — hook it

Each command above runs for minutes (the server cold start has taken ~20 min,
the 4k/16k pair ~50 min, the correctness eval ~16 min). Launch each one
backgrounded with `block_until_ms: 0`, a stable log path, and a
`notify_on_output` hook, then wait with one long `AwaitShell` block. Do not
busy-wait in short polls, and do not use shell `sleep`.

```
block_until_ms: 0
notify_on_output: { pattern: "<see bench-progress-report>", debounce_ms: 60000, reason: "eval progress" }
```

See `bench-progress-report` for the per-job patterns, the wait sizing, and how
to tell a silent stage from a hang.

## Correctness verification

Run this on the **same server and config** as the performance measurement, and
before committing. Level 1 always; add the level that matches the change.

### Level 1 — Task accuracy (always, every candidate)

`math_500` Level 3 via evalscope (command above).

- gate: accuracy ≥ **0.95**; baseline **0.962**
- cost ~16 min, 105 problems, and it needs the server up — budget for it
- read the score from the `Overall report table` in the log, or
  `<work-dir>/<timestamp>/reports/minicpm/math_500.json`
- record the score in the commit body. `--ignore-errors` means a single blown
  request does not fail the run: check `predictions/minicpm/math_500_Level 3.jsonl`
  still has all 105 rows before trusting the number.

### Level 2 — Output equivalence (perf-only changes)

For a change that must not alter numerics — dispatch path, scheduling,
`num_splits`, cudagraph/torch.compile mode, launch parameters, D2H/H2D plumbing
— assert the generated tokens are unchanged, which is much stronger than
accuracy:

```bash
# same prompts, greedy (temperature=0), fixed max_tokens, both arms
# compare token id sequences, not the decoded strings
```

Greedy decode with a fixed seed and identical prompts must produce **identical
token ids**. Report the diff length; a non-empty diff on a perf-only change is a
failure regardless of its length. This is the check several of this repo's
commits rested on.

### Level 3 — Numerical tolerance (new or swapped kernels)

For a kernel-level change (a FlagGems op replacing a vendor op, a fused kernel, a
reduced-precision path), compare against a **float64 reference computed on the
CPU** over the real shapes, and state the tolerance:

- compute the reference in fp64 on CPU, then cast — never let a Triton kernel
  see an fp64 tensor (it will fail to compile)
- report max absolute and relative error for both the candidate and the op it
  replaces; the candidate must be no worse
- never compare a flag's effect using a short prompt: numerics must be checked on
  the shapes the scored workload actually uses

### What blocks the commit

- accuracy below 0.95, or below the same-session baseline
- any token-id diff on a change that claims to be numerically neutral
- a candidate kernel worse in error than the op it replaces
- correctness verified on a different server/config than the performance run

## Measurement rules

These exist because each of them has already produced a wrong conclusion here.

- **Interleaved A/B.** Alternate baseline and candidate. A number from a
  different session or time window is not a baseline — cross-session comparison
  has already produced a false `+7.44%` in this repo.
- **≥3 valid runs.** `RUNS=4`, `SKIP_FIRST=1`, so judge on runs 2–4. Report the
  spread next to the mean.
- **A bimodal result is a finding, not noise.** 4k has been observed to sit in
  either a ~4550 or a ~5300 Total tok/s mode, and the 5038.75 gate falls between
  them. Runs that disagree across the two modes say nothing about the change
  under test — resolve the mode before judging. The mode has so far correlated
  with the count of 10 s windows where prompt and generation throughput are both
  0 while `Running` is non-zero.
- **Warm and cold differs.** `torch.compile` cache warmth moves cudagraph
  capture from ~24 s/graph to ~2–4 s/graph. Compare like with like.
- **Verify what is loaded — including the install *type*.** `pip show
  vllm-plugin-fl` (editable path), `vllm_fl.__file__`, `git log -1`. The tree is
  shared; a branch name is not evidence that the code under test is the code that
  ran. And resolving to a path is not enough on its own: check **which copy is
  live** for every third-party library you edit, because an editable install and a
  static pip install look identical from the source tree:

  ```bash
  python -c "import flag_gems, os; print(os.path.dirname(flag_gems.__file__))"
  ```

  Measured in this image: `vllm_fl` is **editable** (so repo edits take effect)
  but `flag_gems` is a **static pip install** (so edits under
  `/workspace/FlagGems` have *zero* runtime effect — only hand-editing
  `site-packages` works). An optimization validated only against `site-packages`
  has never run on the delivery path, which `pingshen.md` invalidates: line 124
  requires the committee to reproduce the submitted scheme, and line 126 requires
  a PR. Make the source repo editable, or re-verify against the installed copy,
  before claiming a win.
- **Watch for a half-removed install.** `pip uninstall` **skips files whose
  content was modified**, so converting a hand-patched static install to editable
  can leave stale modules behind. Those remnants turn the package into a
  namespace package (`X.__file__ is None`) and shadow the repo version. After the
  switch, assert `X.__file__` points at the repo *and* that no stale directory
  remains in `site-packages`.

## Step 3: commit rules

Commit only when all four metrics improved **and** correctness passed.

```bash
git add <only the files this change touched>
git commit -m "<scope>: <what changed and why>"
```

Include the measured numbers in the commit body: the four deltas, the
correctness score, and the run counts.

- **Never** `git add -A` / `git commit -a`. This repository is shared with
  concurrent sessions; committing someone else's uncommitted work is a real
  failure mode here.
- Commit the change, not the benchmark logs. Keep raw results under
  `/root/bench_results/<tag>/` and reference the path in the commit body.
- If only some metrics improved, do not commit. Record the result, revert or
  park the change on a branch, and return to step 1 with the new information.
- If a change is a win but out of scope for the scored metrics, leave it out of
  the commit rather than mixing concerns.

## Reporting

One table per candidate, then the verdict:

```
[<hh:mm>] <change>
  4k  Total <v> (<d>%)  TTFT <v>ms (<d>%)   <PASS/FAIL>
  16k Total <v> (<d>%)  TTFT <v>ms (<d>%)   <PASS/FAIL>
  correctness  accuracy <score> (gate 0.95)  <PASS/FAIL>
               equivalence <identical N tokens | max|Δ| <tol> | n/a>  <PASS/FAIL>
  spread: 4k <x>% / 16k <y>%   runs: 3 valid
  verdict: <COMMITTED as <sha> | NOT COMMITTED - <which metric failed>>
```

Write to the user in their language; keep the per-metric line to one line.

## Anti-patterns

- Committing because the change "should" help, without both evals.
- Calling a gate-clearing number an improvement when the same-session baseline
  was higher.
- Judging on run 1, on one case, or on a single arm.
- Averaging a bimodal result and declaring a verdict.
- `git commit -a` in a shared worktree.
- Reporting the plan instead of running the gate.
