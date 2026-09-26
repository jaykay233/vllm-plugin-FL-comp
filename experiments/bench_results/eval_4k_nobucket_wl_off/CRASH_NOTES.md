# EngineCore crash (2026-09-25 ~18:41)

## What happened
- Config: no flagos_whitelist (`[]`), FlagGems `align32_geometric` installed, async scheduling ON, `VLLM_GC_DEBUG` unset/0.
- Run 1/4 completed cleanly: 4679.96 tok/s, 256/256 success.
- Mid Run 2 (decode/prefill mix, ~18:41:11 still healthy @1362 gen tok/s):
  APIServer logged `MPClient: engine core exited unexpectedly` at 18:41:21.
- No EngineCore Python traceback. Last EngineCore log line was much earlier
  (18:34:38 attention num_splits). Typical **signal-level / native death**.
- Run 2 reported 6040 tok/s but **128 failed / 128 ok** → invalid.
- dmesg/apport: no useful OOM/segfault record. GPU idle after death.
- Post-crash health: `torch.cuda.Stream()` + matmul OK → device recovered.

## Likely class
Same family as prior MetaX `EngineDeadError` cases in FINDINGS §5.4 /
`server_CRASH.log`: EngineCore child dies hard; parent only sees EngineDeadError.
Not attributable to whitelist-off alone without more stacks; next run enables
`PYTHONFAULTHANDLER=1` to catch fatal signals.
