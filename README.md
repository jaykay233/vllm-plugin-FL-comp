# vllm-plugin-FL

vllm-plugin-FL is a plugin for the [vLLM](https://github.com/vllm-project/vllm) inference/serving framework, built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends vLLM's capabilities and performance across diverse hardware environments. Without changing vLLM's original interfaces or usage patterns, the same command can run model inference/serving on different chips.

## Version Compatibility

| vllm-plugin-FL Branch | Community vLLM Version |
|-----------------------|------------------------|
| `release/0.2` | [v0.20.2](https://github.com/vllm-project/vllm/tree/v0.20.2) |
| `main` | [v0.24.0](https://github.com/vllm-project/vllm/tree/v0.24.0) |

## Supported Models and Chips

In theory, vllm-plugin-FL can support all models available in vLLM, as long as no unsupported operators are involved. The tables below summarize the current support status of end-to-end verified models and chips, including both fully supported and in-progress ("Merging") entries.

### Supported Models

| Model | Status | Reference |
|-------|--------|-----------|
| Qwen3.5-397B-A17B | Supported | [example](./examples/qwen3_5_offline_inference.py) |
| Qwen3-Next-80B-A3B | Supported | [example](./examples/qwen3_next_offline_inference.py) |
| Qwen3-4B | Supported | [example](./examples/offline_inference.py) |
| MiniCPM-o 4.5 | Supported | [example](./examples/minicpm/) |
| GLM-5 | Supported | [example](./examples/glm_5_offline_inference.py) |
| Qwen3.5-35B-A3B | Supported | [example](./examples/qwen3_5_offline_inference.py)  |
| BAAI/bge-m3 | Supported | [implementation](./vllm_fl/models/bge_m3.py) |
| MiniMax-M2.7 | Supported | [implementation](./examples/minimax_m27_offline_inference.py) |

### Supported Chips

| Chip Vendor | Status | Reference |
|-------------|--------|-----------|
| NVIDIA | Supported | - |
| Ascend | Supported | - |
| MetaX | Supported | - |
| T-Head | Supported | - |
| Iluvatar | Supported | - |
| Tsingmicro | Supported | - |
| Moore Threads | Supported | - |
| Hygon | Supported | - |
| Sunrise | Supported | - |

## Quick Start

### Setup

1. Install vLLM

    For **NVIDIA** GPUs, install vLLM from the official [v0.24.0](https://github.com/vllm-project/vllm/tree/v0.24.0) release (optional if the correct version is already installed):
    ```sh
    pip install vllm==0.24.0
    ```

    For **non-NVIDIA** chips, install vLLM from source with the `empty` device target:
    ```sh
    git clone -b v0.24.0 https://github.com/vllm-project/vllm.git
    cd vllm
    VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation --no-deps .
    ```

2. Install vllm-plugin-FL

    2.1 Clone the repository:

    ```sh
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    ```

    2.2 Install
    ```sh
    cd vllm-plugin-FL
    pip install --no-build-isolation .
    # or editable install
    pip install --no-build-isolation -e .
    ```

    For CUDA-like devices, including CUDA and HIP/ROCm environments that use
    PyTorch's CUDA dispatch key, build the plugin native extension by setting
    `VLLM_VENDOR=cuda` during installation:
    ```sh
    cd vllm-plugin-FL
    VLLM_VENDOR=cuda pip install --no-build-isolation .
    # or editable install
    VLLM_VENDOR=cuda pip install --no-build-isolation -e .
    ```

    This builds and installs `vllm_fl._C`, which provides native C++ support
    required by some graph/custom-op paths, especially when vLLM is installed
    with `VLLM_TARGET_DEVICE=empty`.

    If `VLLM_VENDOR` is not set, vllm-plugin-FL is installed as a Python-only
    plugin and the native extension is skipped.

3. Install [FlagGems](https://flagos-ai.github.io/FlagGems/getting-started/install/)

    3.1 Install Build Dependencies

    ```sh
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

    3.2 Install FlagGems

    ```sh
    git clone -b v5.3.4 https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation .
    # or editable install
    pip install --no-build-isolation -e .
    ```

### Runtime compatibility hooks

The plugin installs runtime compatibility hooks through vLLM's plugin entry
points without modifying the installed vLLM package. Model-specific config and
model registrations are loaded only for their corresponding architectures.

Operator adapters use the plugin dispatch manager, so backend selection,
fallback, per-op policy, operator-list recording, and I/O diagnostics continue
to follow the common FlagOS controls.

4. (Optional) Install [FlagCX](https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md#build-and-installation)

    4.1 Clone the repository:
    ```sh
    git clone -b v0.13.0 https://github.com/flagos-ai/FlagCX.git
    cd FlagCX
    git submodule update --init --recursive
    ```

    4.2 Build the library with different flags targeting to different platforms:
    ```sh
    make USE_NVIDIA=1
    ```

    4.3 Set environment
    ```sh
    export FLAGCX_PATH="$PWD"
    ```

    4.4 Installation FlagCX
    ```sh
    cd plugin/torch/
    FLAGCX_ADAPTOR=[xxx] pip install . --no-build-isolation
    # or editable install
    FLAGCX_ADAPTOR=[xxx] pip install -e . --no-build-isolation
    ```
    Note: [xxx] should be selected according to the current platform, e.g., nvidia, ascend, etc.


If there are multiple plugins in the current environment, you can specify use vllm-plugin-fl via VLLM_PLUGINS='fl'.

### Additional Steps for Ascend

1. Install [FlagTree](https://github.com/flagos-ai/flagtree/)

    ```sh
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
    python3 -m pip install flagtree==0.6.1rc1+ascend3.5 $RES
    ```

    For other chips, please refer to [FlagTree](https://github.com/flagos-ai/flagtree/) for the corresponding version (e.g., `flagtree==0.6.1+iluvatar3.6`, `flagtree==0.6.1+metax3.6`, etc.).

2. Set required environment variable

    ```sh
    export TRITON_ALL_BLOCKS_PARALLEL=1
    ```

3. Enable eager execution

    Ascend requires eager execution. Add `enforce_eager=True` to the `LLM` constructor or pass `--enforce-eager` on the command line.


### Run a Task

#### Offline Batched Inference
With vLLM and vLLM-fl installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing). See the example script: [offline_inference](./examples/offline_inference.py). Or use blow python script directly.
```python
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen3-4B", max_num_batched_tokens=16384, max_num_seqs=2048)
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Advanced use

For dispatch environment variable usage, see [environment variables usage](./vllm_fl/dispatch/README.md#environment-variables).

### Using Cuda Communication library
If you want to use the original Cuda Communication, you can unset the following environment variables.
```sh
unset FLAGCX_PATH
```

### Using native CUDA operators
If you want to use the original CUDA operators, you can set the following environment variables.
```sh
export USE_FLAGGEMS=0
```

### CUDAGraph without torch.compile (`VLLM_FL_CUDAGRAPH_ONLY`)

CUDAGraph replay does **not** require `torch.compile`. Only the `PIECEWISE` cudagraph
mode does (`CUDAGraphMode.requires_piecewise_compilation()`), while `FULL` and
`FULL_DECODE_ONLY` replay fine from an eager model. `enforce_eager` cannot be used to
reach that state, because it disables cudagraph and torch.compile **together**.

```sh
export VLLM_FL_CUDAGRAPH_ONLY=1
```

With the flag set, the plugin rewrites the compilation config as
`mode=NONE` + `cudagraph_mode=FULL_DECODE_ONLY` + `custom_ops=['all']`. Dropping
inductor also restores the `custom_ops=['all']` default, so the out-of-tree chain
(`forward_oot -> CachedOp -> flag_gems`) is reachable again for **every** FL op, and
per-op python dispatch no longer costs anything at decode time because the whole step
is one graph replay.

Measured on MetaX C500 / MiniCPM5-2B (paged attention, concurrency 1, median of 7
runs). The first two rows are a same-session A/B on an identical prompt (one run
each, adjacent), so they are directly comparable; `†` marks figures from an earlier
session on a ~500-token prompt that are *not* part of that A/B.

| Configuration | Decode | FlagGems ops reached | Startup |
|---|---|---|---|
| `VLLM_FL_CUDAGRAPH_ONLY=1` | **155.7 tok/s** | rms_norm, silu_and_mul, rotary_embedding | 59 s |
| default (`VLLM_COMPILE` + `FULL_AND_PIECEWISE` + IR bridge) | 147.8 tok/s | rms_norm | 101 s |
| default, `VLLM_FL_IR_KERNELS=0` | 137.7 tok/s † | – | 78 s |
| `enforce_eager=True` (no cudagraph) | 16.4 tok/s † | rms_norm, silu_and_mul, rotary_embedding | 84 s |

So dropping inductor still wins by ~5% even after the IR bridge was optimised to
near-parity, and it reaches three FlagGems ops instead of one. Two mechanisms explain
the gap: `custom_ops=['all']` restores the OOT chain for *every* FL op
(`silu_and_mul` and `rotary_embedding` have no IR-op counterpart, so under
`torch.compile` they are stuck on inductor/torch natives), and the whole decode step
becomes one graph replay, which removes per-op python dispatch
(`CachedOp` -> manager -> resolve) from the hot path entirely.

Notes:

- `PIECEWISE`-only cudagraph modes (and MUSA, which downgrades full graphs to
  `PIECEWISE`) are left untouched: they genuinely need compilation, and the plugin logs
  a warning instead of silently changing behaviour.
- Prefill/mixed batches run eagerly in this mode. If your workload is prefill-heavy,
  compare against the default before adopting it.
- `--compilation-config` / `--enforce-eager` still win; the flag is a default, not an
  override of an explicit user choice.

### FlagGems under torch.compile (`VLLM_FL_IR_KERNELS`)

With the inductor backend vLLM forces `custom_ops=['none']`, so the OOT chain never
runs and `RMSNorm` silently falls back to the inductor-generated kernel. The plugin
registers a `flagos` provider for vLLM's IR ops (`vllm.ir.ops.rms_norm`,
`fused_add_rms_norm`) so FlagGems stays reachable inside `torch.compile(fullgraph=True)`:

```sh
export VLLM_FL_IR_KERNELS=1   # on by default when FlagGems is in use
```

The provider is an opaque `torch.library.custom_op` wrapping
`vllm_fl.dispatch.call_op`, so the dispatch policy, whitelist, fallback and IO-dump
machinery keep working.

`fused_add_rms_norm` is the expensive half of this path, because FlagGems' fused
kernel is in-place while `torch.library.custom_op` forbids returning an aliased
input — so the fused form needs a clone of both activations on every call (2 clones
x 2 calls per layer = 168 extra kernel launches per decode step, ~8.6% of TPOT).
`VLLM_FL_IR_FUSED_ADD` selects the workaround:

```sh
export VLLM_FL_IR_FUSED_ADD=split   # default: plain residual add + functional rms_norm
export VLLM_FL_IR_FUSED_ADD=fused   # old path: FlagGems in-place fused kernel + clones
```

`split` (the default) is clone-free and measured +2.6% throughput over `fused`;
`residual_out` stays bit-identical to vLLM's native result because a bf16 add is
exactly representable in fp32.

Set `VLLM_FL_IR_KERNELS=0` to disable the bridge entirely. If you only want
performance, prefer `VLLM_FL_CUDAGRAPH_ONLY=1` above — it is both faster and reaches more
FlagGems ops.
