# Host-gap measure — 2026-10-23 (P2 / MiniCPM5-2B / 4k)

Official `[[4096,1024,64,256]]` ×2. Env: P2 (GEMV + silu fusion + MBT8192 + cg spare72 + whitelist), `LM_HEAD_GEMM=0`.
Instrumentation: `--enable-logging-iteration-details` + `VLLM_ITER_STAGE_PROFILE=1` (site-packages patch, default-off via env).

## Client (authoritative wall)

| run | duration s | total tok/s | output tok/s | Mean ITL ms | ok |
|-----|------------|-------------|--------------|-------------|-----|
| 1 | 111.64 | 11740.77 | 2348.15 | 25.31 | 256/256 |
| 2 | 111.63 | 11741.84 | 2348.37 | 25.26 | 256/256 |
| **avg** | **111.64** | **11741.3** | 2348.3 | 25.3 | |

## §5.2 closure: iter sum vs client wall

`iteration elapsed` ≈ `future.result` (+ sample inside the timed `with`). From `iter_per_run.py`:

| run | client s | Σ iter s | **non-iter s** | **non-iter %** | iter % |
|-----|----------|----------|----------------|----------------|--------|
| 1 | 111.64 | 91.1 | **20.5** | **18.4%** | 81.6% |
| 2 | 111.63 | 91.6 | **20.0** | **17.9%** | 82.1% |

对比 FINDINGS §5.2 旧数（高峰轮 148.9 / 246.6 → **非迭代 ~40%**）：
**P2 后非迭代占比已降到 ~18%。** 旧 40% 不能再当当前可回收头寸。

轮间唯一 >1s 空档：27s（run1→run2 客户端冷却/tokenize），归因「等请求」，不是引擎停顿。本轮 **无 20s 引擎冻结**。

## Stage-prof 分段（稳态 22 窗 / 221s / 8175 it，`unmarked≈0.11s` 闭合 OK）

| 段 | 合计 s | 占窗口 | ms/it | 含义 |
|----|--------|--------|-------|------|
| **wait** | 181.35 | **82.1%** | 22.18 | GPU 有效等待 |
| exec | 22.22 | 10.1% | 2.72 | `execute_model` 提交（非阻塞 launch） |
| sample | 6.46 | 2.9% | 0.79 | `sample_tokens` 提交 |
| sched | 6.21 | 2.8% | 0.76 | `scheduler.schedule()` |
| upd | 2.77 | 1.3% | 0.34 | aborts + `update_from_output` |
| resid | 2.03 | 0.9% | 0.25 | 入队 + post_step |
| idle | 0.00 | 0.0% | — | 稳态窗内无休眠 |

- **GPU 受限**：busy 里 wait 82%、host 18%。
- Host 最大头：`exec` 提交 ~2.7 ms/it（~10% 墙钟）。这正是 PDL/`MACA_GRAPH_LAUNCH_MODE=2` 想叠的缝；该模式当前 **cudagraph capture 崩**（见 `hostgap_pdl_20261023`），不能开。
- `sched+upd+resid` 合计 ~5%，空间小。
- 与 §5.2 闭合：每轮 non-iter ~20s ≈ `(sched+exec+upd+resid)/2 ≈ 16.6s` 同量级（秒级 iter 墙钟有量化误差）。

## 判定

1. **Host 空档已量清**：当前 P2 上非迭代 ~**18%**（不是 40%）；引擎 busy 内 host ~**18%**，其中 **exec launch ~10%** 是唯一像样的可叠缝头寸。
2. **不要按旧 40% 估收益**；理论把 host 全叠进 GPU 的引擎上限约 `1/0.82 ≈ 1.22×`，实际可动的主要是 exec 那 10%，且依赖 MetaX 修 PDL/recompile。
3. 埋点补丁留在 site-packages，**默认关**（无 `VLLM_ITER_STAGE_PROFILE=1` 零开销）。测完可 `stage_prof_patch.sh revert`。

## 产物

- `bench_4k.log` / `server.log` / `stage_prof.txt` / `launch_serve.py`
- 分析：`experiments/src/iter_per_run.py`、`stage_prof_scan.py`
