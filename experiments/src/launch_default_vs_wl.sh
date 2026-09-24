#!/bin/bash
cp /root/src/run_default_vs_wl.sh /tmp/run_default_vs_wl.$$.sh
exec bash /tmp/run_default_vs_wl.$$.sh "$@"
