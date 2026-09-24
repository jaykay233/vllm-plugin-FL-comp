---
name: self-directed-optimization
description: >-
  Execute the agent's own analysis-derived optimization plan immediately instead
  of ending a turn with a "should I proceed?" question. Use in this repository
  whenever a measurement, profile, or failure has produced a ranked list of next
  steps, when a bottleneck root cause has been identified, when a result has to
  be reproduced or A/B-tested against a baseline, or when the agent would
  otherwise ask the user to choose between options it already recommended.
---

# Self-Directed Optimization

## Core rule

When your own analysis produces a ranked recommendation, that recommendation
**is** the plan. Start executing its top item in the same turn.

> ai后面所有计划都按照自己得出的建议计划进行优化

Do not end a turn with a question whose answer you already wrote down. The
standing instruction in this repository is: derive the plan, then run it.

## The loop

```
1. Derive   rank the next actions; for each, name the metric it should move
            and the baseline it is compared against
2. Execute  do item 1 now — not "want me to?", not a menu of options
3. Verify   measure in the same session against a same-session baseline;
            an unverified change is a hypothesis, not a result
4. Re-derive update the ranking from what you just learned; go to 1
5. Report   results + the re-derived plan; keep going into the next item
```

Stop only to report, never to request permission for a step you already
recommended.

## Already decided — just do it

- the next diagnostic after a surprising measurement
- the next A/B after a root cause is identified
- reading a log or file you just cited
- re-running a measurement that is visibly noisy
- choosing between approaches when one is clearly better here (see below)
- sweeping a parameter whose optimum you just found to be workload dependent

## Still stop and ask

This skill is about not stalling on your own recommendations — not about
override. Ask, with the recommended option first labelled `(Recommended)`:

- **Destructive or irreversible**: killing a process you do not own, deleting
  data or directories, clearing GPU memory, restarting a shared service.
- **Shared resources**: this repository and the GPU are used by concurrent
  sessions. Never kill another session's server or benchmark. Never commit a
  tree containing someone else's uncommitted work.
- **The user said hold**: "先不要测了", "我只是问一问", "停了吧", "don't touch
  the GPU" overrides this skill until they release it. Honouring that is not a
  violation of the loop; ignoring it is.
- **Genuinely blocked**: every approach tried and failed, or the decision is
  theirs alone (scope, priorities, credentials, spending).

## Related skills

- **`eval-gated-optimization`** — the concrete loop for turning a located
  bottleneck into a committed change (verify with the official throughput/TTFT
  benchmark plus the `math_500` correctness eval, then gate the commit on all
  four metrics improving). When item 1 of your plan is a code change, execute it
  through that skill rather than improvising the verification.
- **`bench-progress-report`** — every step of this loop that runs a benchmark,
  eval, sweep, or server start is a multi-minute job. Launch it backgrounded with
  a `notify_on_output` hook and wait with one long `AwaitShell` block; never
  busy-wait. "Keep optimizing" does not mean "poll in 60 s slices".

## Repo-specific rules

These are the mistakes this loop has already produced here. They are
requirements, not suggestions.

- **Interleaved A/B only.** Never compare against a number produced in a
  different session or time window. Cross-session deltas have already produced a
  false `+7.44%` conclusion in this repo. Alternate the arms.
- **Correctness counts as verified, not optional.** A change is not "verified"
  because throughput moved. Verify task accuracy (`math_500` Level 3 ≥ 0.95) plus
  the equivalence level that matches the change — identical token ids for a
  perf-only change, fp64 tolerance for a new kernel. A correctness regression
  blocks the commit even when all four performance metrics improve. See
  `eval-gated-optimization` for the procedure.
- **Reproduce before concluding.** One run that disagrees with a prediction is a
  hypothesis. The harness runs 4 cases and skips run 1 — judge on the remaining
  **≥3 valid runs**, and report the spread next to the mean.
- **A bimodal result is a finding, not noise.** When runs cluster into two
  stable levels, chase the state that separates the modes instead of averaging
  them away. Here that has been worth more than any single micro-optimization.
- **Verify the code that is actually loaded** before attributing a result to a
  change: `pip show` for the editable path, `vllm_fl.__file__` for the import
  that wins, `git log -1` for the revision under test. The tree is shared and a
  branch name is not evidence.
- **Durable artifacts.** Write the plan and its results to `RESULTS.md` (or a
  canvas) so they outlive the turn and can be re-ranked later.

## Report template

One line per executed item:

```
[<hh:mm>] <item> · <metric> <value> (<delta vs baseline>) · <verdict> · <next>
```

Then state the re-derived plan and begin its top item. Write to the user in
their language; keep the per-item line to one line.

## Anti-patterns

- Ending a turn with `要我等它跑完后自动接着查吗？` when the plan is already
  written above it.
- Listing options as prose (letters, numbers, bullets) instead of executing the
  best one or using `AskQuestion` for a decision that is genuinely theirs.
- Treating a held or blocked state as licence to keep pushing.
- Reporting a plan and calling the turn finished.
- Averaging a bimodal or single-run result and declaring a conclusion.
