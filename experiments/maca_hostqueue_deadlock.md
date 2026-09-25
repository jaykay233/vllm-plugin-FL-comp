# MACA 命令队列死锁：`torch.cuda.Stream()` 永久挂死

> **结论**：2026-09-25 这台容器里的 MACA 执行运行时在创建主机命令队列时死锁，
> `torch.cuda.Stream()` 永不返回。这不是本仓库的代码问题，也不是 profiler 配置问题，
> 而是**宿主驱动层的 GPU 上下文泄漏**，**重启容器无法修复**。
> vLLM 必然踩到它：`vllm_fl/worker/model_runner.py:757` 在 async scheduling 开启时
> 无条件 `Stream()`，而 async scheduling **默认就是开的**。
>
> 本文件记录诊断过程、证据、以及为什么排除了其它解释。

脚本：`src/maca_queue_hang_repro.py`（复现）、`src/gpu_health_check.py`（恢复后自检）
原始栈：`stacks_maca_hostqueue.txt`

---

## 1. 现象

`vllm serve` 启动时卡在 `parallel_state.py:1923` 之后、`Loading weights` 之前，
GPU 显存只有 ~826 MiB（权重根本没加载），不报错、不退出。

对照 2026-09-24 正常的一次启动：

```
10:35:14  parallel_state.py:1923
10:35:18  Loading weights took 2.94 seconds      <- 只隔 4 秒
```

2026-09-25 则是：`parallel_state` 之后无限期无进展。

## 2. 环境

| 项 | 值 |
|---|---|
| GPU | MetaX C500（单卡） |
| MACA | 3.7.0.38（驱动 3.8.30） |
| vLLM | 0.24.0 |
| conda env | `mx` |
| 容器 | `de9e46540fc8`，`/sys` 只读挂载，无 `CAP_SYS_ADMIN` |
| 模型 | `/workspace/MiniCPM5-2B`（LlamaForCausalLM，bf16，max_position=131072） |

## 3. 定位过程

因为 `vllm serve` 是父/子进程结构（`EngineCore` 是子进程），父进程的栈只显示它在
`wait_for_engine_startup` 等子进程，看不到根因。改用 `VLLM_ENABLE_V1_MULTIPROCESSING=0`
把引擎放进同进程，再用 `faulthandler.dump_traceback_later` 定时 dump，一次命中：

```
File "torch/cuda/streams.py", line 37 in __new__
File "vllm_fl/worker/model_runner.py", line 757 in __init__
File "vllm_fl/worker/worker.py", line 473 in init_device
```

对应代码就是建流这一句：

```python
if self.use_async_scheduling:
    self.async_output_copy_stream = current_platform.torch_device_fn.Stream()
    self.prepare_inputs_event = torch.Event()
```

`current_platform.torch_device_fn` 就是 `torch.cuda`（`vllm_fl/platform.py:250`），
所以卡点 = **`torch.cuda.Stream()`**。

## 4. 最小复现：设备没坏，只有建流挂

同一进程内逐项测：

| 操作 | 结果 |
|---|---|
| `import torch` | 1.7 s |
| `torch.cuda.is_available()` | **21.1 s**（正常应 < 0.5 s） |
| `torch.cuda.init()` | 完成 |
| `randn(2048,2048,bf16) @ 同上` | **正常，0.8 s**，结果正确 |
| `torch.cuda.current_stream()` | 正常，0 s |
| **`torch.cuda.Stream()`** | **永不返回**（实测 > 13 分钟） |

即：**算力完全正常，唯独「新建流」这一个操作死等。**

### 4.1 反证：底层 API 直调正常

用 `ctypes` 直接调 MACA 运行时（`/opt/maca/lib/libmcruntime.so`）：

```
mcStreamCreate                 -> ret=0  handle=0x...  0.02 s
mcStreamCreateWithFlags        -> ret=0  handle=0x...  0.02 s
mcStreamCreateWithPriority     -> ret=0  handle=0x...  0.02 s   <-- torch 内部用的正是这个
torch.cuda.ExternalStream(handle) -> OK，且能在该流上正常计算
mcGetDevice / mcCtxGetDevice / mcDeviceGetStreamPriorityRange / mcDeviceGetAttribute / mcDeviceGetName
                               -> 全部 ret=0，0.02 s
```

**同一个函数，ctypes 直调 0.02 s，torch 路径死锁** ⇒ 不是 API 坏了，
而是 MACA 执行运行时内部的**锁/信号量没人唤醒**。

## 5. 根因：原生调用栈

装 `py-spy`（宿主的 yama ptrace_scope=1 且容器无 `CAP_SYS_PTRACE`，所以只能用
「py-spy 自己启动目标进程」的方式，且 `--nonblocking` 与 `--native` 不兼容）：

```
py-spy record --format raw --native -d 150 --rate 25 -o out.raw -- \
  python -u patience_stream.py
```

抓到两条关键栈（完整清单见 `stacks_maca_hostqueue.txt`）：

### 5.1 「建流挂死」——死等信号量

```
torch/cuda/streams.py:37                        __new__
  -> THCPStream_pynew
  -> c10::cuda::getStreamFromPool
  -> c10::cuda::initDeviceStreamState -> initSingleStream
  -> wcudaStreamCreateWithPriority          (libruntime_cu.so)
  -> mcStreamCreateWithPriority             (mc_runtime_api.cpp:610)
  -> imcStreamCreate                        (mc_stream.cpp:875)
  -> mcr::Stream::Create                    (mc_stream.cpp:348)
  -> mxr::HostQueue::HostQueue              (mx_commandqueue.cpp:33)   <-- 建主机命令队列
  -> mxr::Monitor::wait                     (monitor.cpp:255)
  -> mxr::Semaphore::timedWait              (semaphore.cpp:88)
  -> futex                                  (libc)                      <-- 永久等待
```

### 5.2 「init 慢 21 秒」——在解压设备码（这个能完成，不是死锁）

```
torch.cuda.init -> mcGetDeviceCount (mc_runtime_api.cpp:227)
  -> std::call_once                          (mutex:697)                <-- 单次初始化
  -> mcr::Init                               (mc_device.cpp:933)
  -> mccInit                                 (mcc_api.cpp:18)
  -> mcc::CODatabase::Init                   (mcc_co_database.cpp:74)
  -> mcc::FatBinaryInfo::ExtractFatBinary
  -> Decompression::decompress -> ZSTD_decompressBlock_internal          <-- MACA 编译器解压内嵌内核
```

采样分布（150 s 窗口，共 634 样本）：

| 栈 | 样本 | 含义 |
|---|---|---|
| `torch.cuda.init()` → `mcc` 解压 | 572 | 慢，但**会结束** |
| `Stream()` → `HostQueue` → `Semaphore` | 5 | **永不结束** |

> ⚠️ 读采样时要注意这点：多数样本落在 init 的 ZSTD 解压上，
> 因为它占满了一个 21 s 的连续区间；而真正致命的 `Stream()` 死锁只有 5 个样本。
> 只看「最高频栈」会误判成「卡在解压」。

## 6. 为什么不是「我杀进程杀坏了」

这是当时最需要排除的怀疑（本次排查中确实对多个 profiler 进程做过 `kill -9`）。
证据链指向**故障早于我的介入，且在容器之外**：

**6.1 时间线**

| 时间(09-25) | 事件 | 依据 |
|---|---|---|
| 15:17:59 | 容器启动（PID 1 `entrypoint.sh`） | `ps -o lstart -p 1` |
| **15:24:04** | `decode_attribution/run.log` **已卡在 `parallel_state`** | 该日志末行 |
| 15:29:26 | `attn_split_sweep.json` 产出真实吞吐 1272 GB/s | 说明当时 GPU 能算 |
| **15:31:45** | `run2.log` 又卡在 `parallel_state` | 该日志末行 |
| 15:39+ | 本会话第一次起进程 | — |

15:24 与 15:31 两次挂死的表现与现在**完全一致**，而它们都早于我 15:39 的第一次操作。

**6.2 容器重启无效（决定性）**

17:12:22 容器整体重建。重启后立刻复测：

```
import torch       1.7 s
is_available=True  21.1 s
Stream()           挂死（退出码 124）
```

**如果是我的进程造成的，容器重建必然清掉。** 重启后不仅没修好，
`mx-smi` 的 826 MiB 幽灵占用还照样在。

**6.3 幽灵占用：无进程却占显存**

```
$ mx-smi
| 826/65536 MiB | Available |
| Process:                |
|   no process found      |

$ mx-smi --show-all-process      # 宿主视角
|  no process found      |

$ mx-smi --show-unavailable-reason
GPU#0  MXC500  0000:0e:00.0   available
```

扫描全机所有 `/proc/*/fd`，**没有任何进程持有 GPU 设备节点**。
即：826 MiB 是驱动内核侧未回收的残留上下文 —— 正是「队列池被占死」的来源，
而它在宿主驱动里，容器重启带不走。

**6.4 这不是新问题，只是恶化了**

`FINDINGS.md` §6.1 在 **2026-09-23** 就记录过：

> stream 创建 | **很慢**（实测最慢 86 s，含重试）

也就是说「建流慢」在这台机器上是**慢性病**，9/25 只是跨过了临界点变成永久死锁。
这从源头排除了「9/25 的某次操作引入」的解释。

## 7. 容器内无法修复

```
/.dockerenv 存在                        -> 在 Docker 容器内
CapEff: a80425fb                        -> 没有 CAP_SYS_ADMIN（也没有 CAP_SYS_PTRACE）
sysfs on /sys type sysfs (ro,...)        -> /sys 只读挂载
$ mx-smi -i 0 -r -y
GPU#0 reset Sysfs error: Read-only file system     <- reset 走 sysfs，必然失败
$ mount -o remount,rw /sys
mount: /sys: permission denied
```

`mx-smi` 的 `-r/--reset`、`--flr`、`--vfflr` 全部走 sysfs，容器内都不可用。
**必须在宿主层 reset 或重启宿主机。**

## 8. 可用绕过（非根治）

```
vllm serve ... --enforce-eager --no-async-scheduling
```

**能正常启动并服务**（实测 90 s 就绪，`Loading weights took 2.62 seconds`，
`GPU KV cache size: 1,192,592 tokens`）。原因：它同时绕开了两个建流点 ——
`--no-async-scheduling` 跳过 `model_runner.py:757`，
`--enforce-eager` 跳过 CUDA graph capture 的
`GraphCaptureContext(Stream(device=device))`（`model_runner.py:140`）。

⚠️ **但这不能用于评测**：评测口径是 CUDA graph 模式，而 graph capture 必须建流。
所以绕过方案只能用来做「非 graph 的临时验证」，拿不到目标数据。

## 9. 对本次任务的影响

原目标是「用 torch profiler 看真实 serving（CUDA graph）路径的耗时分布」。
该目标被本故障完全阻塞：profiler 连引擎都起不来。
脚本与配置本身已就绪（`src/run_serve_profile.sh`、
`src/run_after_recovery.sh`：先自检 `Stream()`，健康才启动 profiler），
**GPU 恢复后可立即产出 `profiler_out_0.txt`。**

## 10. 复用清单

| 文件 | 用途 |
|---|---|
| `src/maca_queue_hang_repro.py` | 最小复现：`--health` 快速判定，默认输出 init 计时 + 算力证明 + `Stream()` |
| `src/gpu_health_check.py` | 恢复后自检，healthy 才继续跑压测 |
| `stacks_maca_hostqueue.txt` | `py-spy --native` 原始栈（按采样数排序，关键栈强制保留） |
| `src/run_after_recovery.sh` | 宿主 reset 后一键：自检 -> 起 profiler |

### 排查手法备忘

- **进程间看不到根因时把引擎拉进同进程**：`VLLM_ENABLE_V1_MULTIPROCESSING=0`
  + `faulthandler.dump_traceback_later()`，比在父子进程间猜快得多。
- **`py-spy` 在容器里受限**：`ptrace_scope=1` 且无 `CAP_SYS_PTRACE` 时
  `py-spy dump --pid` 报 `Permission denied`；改成让 py-spy **自己启动目标进程**
  （`py-spy record ... -- python target.py`）即可，`--native` 拿到 C++ 栈是关键。
- **`--nonblocking` 与 `--native` 互斥**，别一起用。
- **看到「最高频栈」先别下结论**：耗时区间（能结束的 21 s 解压）会压过
  真正致命的死锁栈（5 个样本）。要按**是否返回**而不是**样本多少**来分主次。
