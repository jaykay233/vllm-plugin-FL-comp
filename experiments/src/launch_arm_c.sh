#!/bin/bash
cp /root/src/run_arm_c.sh /tmp/run_arm_c.$$.sh
exec bash /tmp/run_arm_c.$$.sh "$@"
