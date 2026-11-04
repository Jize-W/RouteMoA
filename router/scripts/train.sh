#!/usr/bin/env bash
# Usage:
#     bash scripts/train.sh                        # λ=0.5, α=0.2
#     LAMBDA=0.3 ALPHA=0.8 bash scripts/train.sh
#     TRAINING_STEPS=20 bash scripts/train.sh      # quick smoke test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
BACKBONE=${BACKBONE:-microsoft/mdeberta-v3-base}

LAMBDA=${LAMBDA:-0.5}
ALPHA=${ALPHA:-0.2}
RUN_DIR=${RUN_DIR:-runs/lambda_${LAMBDA}_alpha_${ALPHA}}

BATCH_SIZE=${BATCH_SIZE:-64}
TRAINING_STEPS=${TRAINING_STEPS:-350}
EVAL_STEPS=${EVAL_STEPS:-50}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-500}
TOP_K=${TOP_K:-2}
LAST_K=${LAST_K:-2}
H=${H:-3}
SEED=${SEED:-42}
MODEL_ORDER=${MODEL_ORDER:-sorted}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}

DATA_DIR="data/clustered/train_lambda_${LAMBDA}"
DEV_DIR="data/dev_lambda_${LAMBDA}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] training data not found: $DATA_DIR"
  echo "        run first: LAMBDAS=\"$LAMBDA\" bash scripts/prepare_data.sh"
  exit 1
fi
if [[ ! -d "$DEV_DIR" ]]; then
  echo "[error] dev data not found: $DEV_DIR"
  echo "        run first: LAMBDAS=\"$LAMBDA\" bash scripts/prepare_data.sh"
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "=================================================================="
echo " train scorer"
echo "   λ (data blend)     : $LAMBDA    -> $DATA_DIR"
echo "   α (cluster loss)   : $ALPHA"
echo "   batch / steps      : $BATCH_SIZE / $TRAINING_STEPS (eval every $EVAL_STEPS)"
echo "   lr / samples       : $LEARNING_RATE / $TRAIN_SAMPLES per dataset"
echo "   model_order        : $MODEL_ORDER      (sorted=consistent with ckpt/router_small.pth)"
echo "   gradient ckpt      : $GRADIENT_CHECKPOINTING   (1=saves memory, numerically identical)"
echo "   backbone           : $BACKBONE"
echo "   save_path          : $RUN_DIR"
echo "=================================================================="

$PYTHON src/train_onescore.py \
  --data_dir "$DATA_DIR" \
  --test_data_dir "$DEV_DIR" \
  --save_path "$RUN_DIR" \
  --backbone "$BACKBONE" \
  --batch_size "$BATCH_SIZE" \
  --training_steps "$TRAINING_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --training_samples_per_dataset "$TRAIN_SAMPLES" \
  --top_k "$TOP_K" --last_k "$LAST_K" --H "$H" \
  --sample_loss_weight 0 \
  --cluster_loss_weight "$ALPHA" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --model_order "$MODEL_ORDER" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_DIR/train.log"

echo
# Note: best_model.pth is selected by validation performance at the evaluation points (every EVAL_STEPS steps).
# If EVAL_STEPS > TRAINING_STEPS, evaluation never fires during the run, so there is no best checkpoint.
if [[ -f "$RUN_DIR/best_model.pth" ]]; then
  echo "training finished: $RUN_DIR/best_model.pth"
  echo "next, evaluate: RUN_DIR=$RUN_DIR bash scripts/eval.sh"
else
  echo "[warn] best_model.pth was not created: EVAL_STEPS($EVAL_STEPS) is larger than TRAINING_STEPS($TRAINING_STEPS),"
  echo "       so evaluation never fired and there is no checkpoint to select. Set EVAL_STEPS <= TRAINING_STEPS."
  exit 1
fi
