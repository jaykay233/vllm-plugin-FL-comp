#!/bin/bash
cp /root/src/run_mm_strategy_ab.sh /tmp/run_mm_strategy_ab.$$.sh
exec bash /tmp/run_mm_strategy_ab.$$.sh "$@"
