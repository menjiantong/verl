#!/bin/bash
# Kill leftover ray/vllm/verl processes. Kept in a file so the pkill pattern never matches the
# invoking shell's own command line (pkill -f matches full argv, and an inline command that
# *contains* the pattern kills itself).
for pat in 'ray::' 'raylet' 'gcs_server' 'vllm_async' 'vLLMHttpServer' 'python3 -m verl' 'EngineCore' 'TaskRunnerV1'; do
    pkill -9 -f "$pat" >/dev/null 2>&1 || true
done
sleep 2
exit 0