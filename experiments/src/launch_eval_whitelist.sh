#!/bin/bash
# 把评测脚本拷到 /tmp 再执行：bash 是增量读取脚本文件的，
# 若运行期间源脚本被编辑，可能读到半截命令（已踩过一次，见 FINDINGS）。
set -u
SRC=/root/src/run_eval_whitelist.sh
TMP=/tmp/run_eval_whitelist.$$.sh
cp "$SRC" "$TMP"
exec bash "$TMP"
