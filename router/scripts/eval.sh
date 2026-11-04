#!/usr/bin/env bash
# Usage:
#     bash scripts/eval.sh                                   # evaluate runs/lambda_0.5_alpha_0.2
#     RUN_DIR=runs/xxx bash scripts/eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
BACKBONE=${BACKBONE:-microsoft/mdeberta-v3-base}
RUN_DIR=${RUN_DIR:-runs/lambda_0.5_alpha_0.2}
TEST_DIR=${TEST_DIR:-data/int_score/test}
MODEL_ORDER=${MODEL_ORDER:-auto}

WEIGHTS="$RUN_DIR/best_model.pth"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "[error] checkpoint not found: $WEIGHTS"
  echo "        run first: RUN_DIR=$RUN_DIR bash scripts/train.sh"
  exit 1
fi

echo "=================================================================="
echo " evaluate scorer"
echo "   weights : $WEIGHTS"
echo "   test    : $TEST_DIR"
echo "   order   : $MODEL_ORDER"
echo "=================================================================="

$PYTHON src/eval_onescore_direct.py \
  --trained_router_path "$WEIGHTS" \
  --backbone_path "$BACKBONE" \
  --test_data_dir "$TEST_DIR" \
  --model_order "$MODEL_ORDER" \
  --top_k 3 \
  --threshold 1.0 \
  2>&1 | tee "$RUN_DIR/eval.log"

echo
echo "metrics written to $RUN_DIR/eval.log"
