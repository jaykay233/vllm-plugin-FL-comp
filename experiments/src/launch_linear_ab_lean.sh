#!/bin/bash
cp /root/src/run_linear_ab_lean.sh /tmp/run_linear_ab_lean.$$.sh
exec bash /tmp/run_linear_ab_lean.$$.sh "$@"
