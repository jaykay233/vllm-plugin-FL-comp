#!/bin/bash
cp /root/src/run_confirm_lean.sh /tmp/run_confirm_lean.$$.sh
exec bash /tmp/run_confirm_lean.$$.sh "$@"
