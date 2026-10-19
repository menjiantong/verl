#!/bin/bash
cd /workspace-verl/verl
setsid bash /tmp/run_grpo_bi.sh > /tmp/grpo_bi_wrapper.log 2>&1 < /dev/null &
