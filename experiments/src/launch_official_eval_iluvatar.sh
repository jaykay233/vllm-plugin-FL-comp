#!/bin/bash
# 把评测脚本拷到 /tmp 再执行：bash 是增量读取脚本文件的，
# 若运行期间源脚本被编辑，可能读到半截命令（已踩过一次，见 FINDINGS-mx.md）。
#
# 沐曦版对应 launch_eval_whitelist.sh；本文件是天数 BI-V150 的入口。
set -u
SRC=/workspace/vllm-plugin-FL-comp/experiments/src/run_official_eval_iluvatar.sh
TMP=/tmp/run_official_eval_iluvatar.$$.sh
cp "$SRC" "$TMP"
exec bash "$TMP"
