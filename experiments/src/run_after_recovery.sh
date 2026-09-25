#!/bin/bash
# Post-recovery entry point.
#
# The MACA host-queue deadlock (see maca_hostqueue_deadlock.md) blocks
# torch.cuda.Stream(), which blocks CUDA-graph capture, which blocks the scored
# eval path. It can only be cleared from the host (mx-smi -r on the host, or a
# host reboot) -- a container restart does NOT help.
#
# This script therefore gates on a health check first: it refuses to touch the
# model/server until Stream() demonstrably returns, so a still-broken GPU fails
# fast instead of producing another 20-minute silent hang.
#
# Usage:
#     bash run_after_recovery.sh
#
# Override where the profiler driver lives if it is not the default:
#     SERVE_PROFILE_SH=/path/to/run_serve_profile.sh bash run_after_recovery.sh
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D=/root/bench_results/recovery_check
mkdir -p "$D"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

say "1) GPU 自检（关键：torch.cuda.Stream() 是否恢复）"
if timeout 90 python -u "$HERE/gpu_health_check.py" 2>&1 | tee "$D/gpu_check.log" | sed 's/^/     /'; then
  :
fi

if ! grep -q "GPU_HEALTHY" "$D/gpu_check.log"; then
  say "   !! GPU 仍未恢复：Stream() 依旧挂死。"
  say "   宿主层 reset 未生效，或仍有进程占着 GPU。"
  say "   先确认 mx-smi 的幽灵占用已清零（no process found），再重试。"
  exit 2
fi

SERVE_PROFILE_SH="${SERVE_PROFILE_SH:-/root/src/run_serve_profile.sh}"
if [ ! -f "$SERVE_PROFILE_SH" ]; then
  say "   GPU 已恢复，但找不到 profiler 驱动脚本：$SERVE_PROFILE_SH"
  say "   设置 SERVE_PROFILE_SH=<path> 后重跑。"
  exit 3
fi

say "   GPU 已恢复，启动真实 serving 路径（CUDA graph）profiler"
exec bash "$SERVE_PROFILE_SH"
