#!/bin/bash
# 把脚本复制到 /tmp 再执行：避免 Bash 增量读脚本时被编辑打断
cp /root/src/run_linear_ab.sh /tmp/run_linear_ab.$$.sh
exec bash /tmp/run_linear_ab.$$.sh "$@"
