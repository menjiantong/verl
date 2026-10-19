#!/bin/bash
cd /workspace-verl/verl
setsid bash scripts/dsv41_consistency/run_dump_engine.sh --lengths 64 --skip-params > /tmp/dump_engine_router.log 2>&1 < /dev/null &
