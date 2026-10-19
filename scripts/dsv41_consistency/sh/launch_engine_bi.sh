#!/bin/bash
cd /workspace-verl/verl
setsid bash scripts/dsv41_consistency/run_dump_engine.sh --lengths 200 --skip-params --batch-invariant > /tmp/dump_engine_bi.log 2>&1 < /dev/null &
