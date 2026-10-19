#!/bin/bash
cd /workspace-verl/verl
setsid bash /tmp/run_dump_trainer_fp32.sh > /tmp/dump_trainer_fp32.log 2>&1 < /dev/null &
