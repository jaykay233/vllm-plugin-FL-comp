#!/bin/bash
cp /root/src/run_confirm_mm_fix.sh /tmp/run_confirm_mm_fix.$$.sh
exec bash /tmp/run_confirm_mm_fix.$$.sh "$@"
