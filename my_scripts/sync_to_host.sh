#!/usr/bin/env bash
# Mirror the working trees from this machine (131) to a partner machine (SFT/RL peer).
# No rsync in the image -- tar over ssh instead (same convention as sync_to_128.sh).
# Run after every edit batch; the md5 check at the end is the acceptance line.
#
#   bash scripts/sync_to_host.sh <host-ip> [port]      # e.g. bash scripts/sync_to_host.sh 80.5.25.119
#
# Carries exactly what the GRPO/SFT flows import at run time on the peer:
#   scripts/                              launch wrappers + data prep (core dumps excluded)
#   FSDPTurbo/                            trainer engine (rebase 31fd979 = fused-op main + our delta)
#   vllm-ascend-v41-private/vllm_ascend   rollout plugin INCLUDING the rebuilt vllm_ascend_C .so
#   vllm-ascend-v41-private/pyproject.toml
#   vllm/vllm                             the vllm source tree that sits on PYTHONPATH
#   vllm/pyproject.toml
#   verl/verl  verl/examples/grpo_trainer  verl/scripts  verl/wiki
# Excluded: .git (peer keeps its own), __pycache__, *.pyc, core_* (scripts/core_* is a 2.5 GB dump).
# NOT carried: /mnt/share artifacts (already shared) and per-node logs/outputs.
set -eo pipefail
DEST_HOST=${1:?usage: sync_to_host.sh <host-ip> [port]}
PORT=${2:-50001}
DEST="root@${DEST_HOST}"
PATHS=(
  scripts
  FSDPTurbo
  vllm-ascend-v41-private/vllm_ascend
  vllm-ascend-v41-private/pyproject.toml
  vllm/vllm
  vllm/pyproject.toml
  verl/verl
  verl/examples/grpo_trainer
  verl/scripts
  verl/wiki
)
echo "[sync] -> $DEST:$PORT"
echo "[sync] paths: ${PATHS[*]}"
tar czf - -C /workspace-verl \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='core_*' \
  "${PATHS[@]}" \
  | ssh -p "$PORT" -o StrictHostKeyChecking=no "$DEST" 'tar xzf - -C /workspace-verl'
echo "[sync] unpacked. md5 acceptance:"
FILES=(
  scripts/run_grpo_2node.sh
  scripts/run_grpo_top64.sh
  FSDPTurbo/fsdp_turbo/models/deepseek_v41/experts.py
  FSDPTurbo/fsdp_turbo/ops/cpu/sparse_attention.py
  FSDPTurbo/fsdp_turbo/models/deepseek_v41/adapter.py
  vllm-ascend-v41-private/vllm_ascend/attention/dsa_v41.py
  vllm-ascend-v41-private/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
  vllm/vllm/__init__.py
  verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py
)
fail=0
for f in "${FILES[@]}"; do
  a=$(md5sum "/workspace-verl/$f" | cut -d' ' -f1)
  b=$(ssh -p "$PORT" -o StrictHostKeyChecking=no "$DEST" "md5sum /workspace-verl/$f 2>/dev/null | cut -d' ' -f1")
  if [ "$a" = "$b" ]; then echo "  OK   $f"; else echo "  DIFF $f ($a vs '$b')"; fail=1; fi
done
if [ "$fail" = 0 ]; then echo "[sync] done."; else echo "[sync] FAILED -- peer differs"; exit 1; fi
