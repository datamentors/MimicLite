#!/bin/bash
# MimicLite PPO for A2 (single stage, as the paper's Small/Base/Huge). Run inside the container.
# Env: MODULE small|base|large|huge (default small)  NPROC (default: visible GPUs)
#      NUM_ENVS per GPU (default 8192)  ITERS (default 4000)  RUN_NAME (default a2_ppo_<module>_<ts>)
#      EXTRA="hydra overrides ..."  WANDB_API_KEY (online if set, else offline)  WANDB_PROJECT
set -euo pipefail
cd /opt/aa
MODULE=${MODULE:-small}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
RUN=/runs/${RUN_NAME:-a2_ppo_${MODULE}_$(date +%Y%m%d_%H%M%S)}
# W&B project per run type: MimicLiteA2-SmallPPO / -BasePPO / -LargePPO / -HugePPO
RUNTYPE="${MODULE^}PPO"
WANDB_PROJECT=${WANDB_PROJECT:-MimicLiteA2-$RUNTYPE}
if [ -z "${WANDB_MODE:-}" ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then WANDB_MODE=online; else WANDB_MODE=offline; fi
fi
mkdir -p "$RUN"
export WANDB_PROJECT
echo "run=$RUN module=$MODULE nproc=$NPROC envs/gpu=${NUM_ENVS:-8192} iters=${ITERS:-4000} wandb=$WANDB_PROJECT ($WANDB_MODE)" | tee "$RUN.log"
# Keep only latest + best checkpoint while training (and once more at the end).
python /opt/a2/tools/prune_checkpoints.py "$RUN" --watch 600 --stop-file "$RUN/.train_done" >> "$RUN.log" 2>&1 &
PRUNER=$!
trap 'touch "$RUN/.train_done"; kill $PRUNER 2>/dev/null || true' EXIT
python -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29517}" \
    projects/mimic-lite/scripts/train.py \
    task=tracking-base task/robot=a2 backend=mjlab \
    +exp=ppo/train algo/ppo/module="$MODULE" \
    task.num_envs=${NUM_ENVS:-8192} total_iters=${ITERS:-4000} \
    wandb.mode="$WANDB_MODE" wandb.project="$WANDB_PROJECT" wandb.name="$(basename "$RUN")" \
    hydra.run.dir="$RUN" ${EXTRA:-} 2>&1 | tee -a "$RUN.log"
touch "$RUN/.train_done"
kill $PRUNER 2>/dev/null || true
python /opt/a2/tools/prune_checkpoints.py "$RUN" 2>&1 | tee -a "$RUN.log"
