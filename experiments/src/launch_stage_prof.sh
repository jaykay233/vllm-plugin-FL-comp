#!/bin/bash
# 把 run_stage_prof.sh 复制到 /tmp 再执行。
#
# 原因：bash 是增量读取脚本文件的。若在脚本运行期间编辑源文件，bash 会从
# 旧的字节偏移量继续读取被改动过的内容，于是执行到错位的碎片命令。实测后果
# 是 `line 61: ils: command not found`，并且碎片里的 `> "$LOG"` 把 server.log
# 截断成一行报错，整整一轮 26 分钟白跑。副本执行则与源文件编辑互不影响。
set -u
SRC=/root/src/run_stage_prof.sh
DST=/tmp/run_stage_prof.$$.sh
cp "$SRC" "$DST" || exit 1
trap 'rm -f "$DST"' EXIT
exec bash "$DST"
