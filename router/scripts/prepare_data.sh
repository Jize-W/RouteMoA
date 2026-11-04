#!/usr/bin/env bash
# Usage:
#     bash scripts/prepare_data.sh                  # rebuild only what is missing (default λ=0.5)
#     LAMBDAS="0.1 0.3 0.5 0.7 0.9" bash scripts/prepare_data.sh
#     FORCE=1 bash scripts/prepare_data.sh          # force rebuild
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
LAMBDAS=${LAMBDAS:-"0.5"}
BACKBONE=${BACKBONE:-microsoft/mdeberta-v3-base}
N_CLUSTERS=${N_CLUSTERS:-6}
NUM_PER_DATASET=${NUM_PER_DATASET:-2000}
FORCE=${FORCE:-0}

echo "=================================================================="
echo " data preparation"
echo "   λ            : $LAMBDAS"
echo "   backbone     : $BACKBONE"
echo "   n_clusters   : $N_CLUSTERS   (max $NUM_PER_DATASET records per dataset)"
echo "   output       : data/{merged,clustered,dev}"
echo "=================================================================="

base_lambda=""

for lam in $LAMBDAS; do
  merged="data/merged/train_lambda_${lam}"
  clustered="data/clustered/train_lambda_${lam}"
  dev="data/dev_lambda_${lam}"

  if [[ -f "$merged/math-train.json" && $FORCE -eq 0 ]]; then
    echo "[skip] $merged already exists (FORCE=1 to rebuild)"
  else
    echo
    echo "---- λ=$lam : merge (int_score + reward_score) ----"
    $PYTHON src/merge_scores.py \
      --weight "$lam" \
      --int_dir data/int_score/train \
      --reward_dir data/reward_score/train \
      --out_dir "$merged"
  fi

  if [[ -f "$dev/math-dev.json" && $FORCE -eq 0 ]]; then
    echo "[skip] $dev already exists (FORCE=1 to rebuild)"
  else
    echo
    echo "---- λ=$lam : merge (dev split) ----"
    $PYTHON src/merge_scores.py \
      --weight "$lam" \
      --int_dir data/int_score/test \
      --reward_dir data/reward_score/test \
      --out_dir "$dev"
  fi

  if [[ -f "$clustered/math-train.json" && $FORCE -eq 0 ]]; then
    echo "[skip] $clustered already exists (FORCE=1 to rebuild)"
  elif [[ -z "$base_lambda" ]]; then
    echo
    echo "---- λ=$lam : cluster (t-SNE + k-means, run only once) ----"
    $PYTHON src/cluster_generate.py \
      --data_dir "$merged" \
      --out_dir "$clustered" \
      --backbone "$BACKBONE" \
      --n_clusters "$N_CLUSTERS" \
      --num_per_dataset "$NUM_PER_DATASET"
    base_lambda="$lam"
  else
    echo
    echo "---- λ=$lam : inherit cluster_id (from λ=$base_lambda, clustering is independent of λ) ----"
    $PYTHON src/copy_cluster_ids.py \
      --cluster_dir "data/clustered/train_lambda_${base_lambda}" \
      --merged_dir "$merged" \
      --out_dir "$clustered"
  fi

  [[ -z "$base_lambda" ]] && base_lambda="$lam"
done

echo
echo "=================================================================="
echo " done. directory overview:"
for d in data/merged data/clustered data/dev_lambda_*; do
  [[ -d "$d" ]] && printf "   %-40s %s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo " next: bash scripts/train.sh"
echo "=================================================================="
