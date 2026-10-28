#!/usr/bin/env bash
# Mirror the working trees from this machine (131) to the SFT/RL partner machine (128).
# No rsync in the image -- tar over ssh instead. Run after every edit batch; the
# md5 check at the end is the acceptance line (all OK = both machines identical).
set -eo pipefail
DEST=${DEST:-root@80.5.25.128}
PORT=${PORT:-50001}
PATHS=(
  scripts
  verl/wiki
  FSDPTurbo/fsdp_turbo
  verl/verl/workers
  verl/verl/trainer
  verl/verl/utils
  verl/verl/experimental
  verl/examples/grpo_trainer
)
echo "[sync] packing: ${PATHS[*]}"
tar czf - -C /workspace-verl --exclude='__pycache__' --exclude='core_*' "${PATHS[@]}" \
  | ssh -p "$PORT" -o StrictHostKeyChecking=no "$DEST" 'tar xzf - -C /workspace-verl'
echo "[sync] packed+unpacked. verifying md5 of the hot files:"
for f in FSDPTurbo/fsdp_turbo/models/deepseek_v41/model.py \
         verl/verl/workers/engine/fsdp/fsdp_turbo_dsv41_impl.py \
         scripts/train_engine_smoke.py scripts/run_grpo_2node.sh; do
  a=$(md5sum /workspace-verl/$f | cut -d' ' -f1)
  b=$(ssh -p "$PORT" -o StrictHostKeyChecking=no "$DEST" "md5sum /workspace-verl/$f | cut -d' ' -f1")
  [ "$a" = "$b" ] && echo "  OK   $f" || { echo "  DIFF $f"; exit 1; }
done
echo "[sync] done."
