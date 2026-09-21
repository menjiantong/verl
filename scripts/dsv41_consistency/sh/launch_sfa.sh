#!/bin/bash
cd /workspace-verl/verl
setsid bash /tmp/run_dump_trainer_sfa.sh > /tmp/dump_trainer_sfa3.log 2>&1 < /dev/null &
