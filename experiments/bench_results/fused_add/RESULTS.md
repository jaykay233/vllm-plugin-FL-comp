# Killing the 168 decode-time clones in `fused_add_rms_norm`

## TL;DR

`vllm_fl/ops/ir_kernels.py` cloned both activations on **every** call of the
`fused_add_rms_norm` IR op, purely to satisfy `torch.library.custom_op`'s
no-aliasing rule.  At decode that is 168 kernel launches and ~477 us per step,
i.e. **8.6% of TPOT**, all of it launch overhead on 4 KB tensors.

Splitting the operation -- a plain add for the residual, then the already
functional `rms_norm` op -- removes every clone and is still bit-exact:

| | tok/s | us/token | copy launches/step | copy us/step |
|---|---|---|---|---|
| `fused` (before) | 144.2 / 143.2 | 6937 / 6982 | 214.7 / 214.7 | 622.0 / 619.9 |
| **`split` (after)** | **147.7 / 147.5** | **6773 / 6779** | **45.3 / 45.3** | **143.8 / 144.2** |

**+2.6% throughput / -2.6% latency**, reproducible to 0.14% across interleaved runs.

## How the copies were found

Kernel attribution of a MATH-500-shaped decode step (49-token prompt, 1024-token
output, batch=1) showed two copy kernels:

    _copy_kernel_kernel_rank_2   168/step  (4 per layer)  2.66 us/launch   447 us
    _copy_kernel_kernel_rank_3    42/step  (1 per layer)  3.23 us/launch   136 us

168 = 42 layers x 4, and the per-launch cost is pure overhead: the tensor is
`[1, 2048]` bf16 = **4 KB**, which moves in ~0 us.

The MACA profiler records **no python stacks** for CUDA kernels (`with_stack=True`
returns `<no stack>`), and `cpu_parent` is empty, so attribution came from
matching per-step CPU op counts under eager launching:

    aten::clone   168/step   input shape [1, 2048]      <- the copy source
    aten::copy_   168/step   [1, 2048] -> [1, 2048]
    aten::mm      168/step   (4 projections x 42 layers)

`clone` == `mm` == 168/step, and `fused_add_rms_norm` is called 84 times/step, so
**2 clones per call**.  That lands exactly on:

```python
@torch.library.custom_op("vllm_fl::fused_add_rms_norm", mutates_args=())
def fused_add_rms_norm_op(x, x_residual, weight, epsilon, variance_size=None):
    out, residual = _dispatch_rms_norm(x.clone(), x_residual.clone(), weight, epsilon)
    return out, residual
```

## Why the clones were there

FlagGems' fused kernel is in-place: it writes `x + residual` into `residual`,
`rms_norm(x + residual) * w` into `x`, and returns those same tensors.  But
`torch.library.custom_op` forbids an output aliasing an input:

> The output of this custom operator (1) must not also be an input to this custom
> operator and (2) may not alias any inputs to this custom operator or other
> returns. ... Please instead return a clone of the offending output tensor(s).

So the wrapper cloned to stay "functional".  (Re-declaring the mutated args and
returning the same tensors is **not** an escape: that combination raises the exact
error above, verified.)

## The fix

Don't fuse.  Compute the residual sum with a plain, inductor-fusable add and call
the *functional* `rms_norm` op, which already existed and never cloned:

```python
residual_out = x + x_residual                  # new tensor, no clone
out = rms_norm_op(residual_out, weight, eps)   # new tensor, no clone
return out, residual_out
```

`residual_out` is bit-identical to vLLM's native result: summing two bf16 values
is exact in fp32 and rounding back to bf16 is the same correctly-rounded value a
direct bf16 add produces.

Selected by `VLLM_FL_IR_FUSED_ADD` (`split` default, `fused` = old behaviour).
Cost is one extra elementwise add kernel, and the arithmetic is consistent --
the copy saving is 477 us/step while TPOT drops 190 us, the difference being that
add.

## Note: `inplace=` was a red herring

An earlier hypothesis blamed `register_impl(..., inplace=True)`, since vLLM's
`VllmIRLoweringPass` lowers through `IrOpImpl.func_impl_fn`, which clones all
`activation_indices` for inplace impls.  A/B disproved it: `clone_inplace` and
`clone_func` both measured **214.7** copy launches/step.  Also, the `inplace`
clones would be *free* under cudagraph (the donor pattern), whereas these 168
kernels were real.  The custom op's own clones were the whole story.

## Pitfalls

* Kernel `device_time` from `torch.profiler` is the only trustworthy metric for
  ~3 us kernels; CUDA events mostly measure launch overhead.
* `VLLM_ENABLE_V1_MULTIPROCESSING=0` is mandatory for in-process profiling --
  without it the engine runs in a child process and the parent profiler reports
  zero kernels (an early run reported `COPY TOTAL: 0.0` for this reason).
* Leftover background loops from earlier experiments will silently hold GPU
  memory and make the *next* vLLM start fail with "Free memory ... is less than
  desired GPU memory utilization", or skew timings.  Check with
  `ps -eo pid,cmd | grep -E '[c]udagraph|[E]ngineCore'` before measuring.
* Retry logic that greps a shared append-only log will match the *previous*
  run's success marker and report a failed run as OK.  Tail the log or use a
  per-attempt marker.

## Files

* `vllm_fl/ops/ir_kernels.py` -- the fix (`VLLM_FL_IR_FUSED_ADD`).
* `/root/src/trace_copy_kernels.py` -- per-kernel counts + call sites.
* `/root/src/trace_copy_source.py` -- CPU op attribution and tensor shapes.
* `/root/src/copy_by_mode.py` -- copy budget + throughput per configuration.
* `/root/bench_results/fused_add/run.log`, `/root/bench_results/fused_add2/run.log`
  -- raw A/B logs.
