#!/usr/bin/env bash
# Usage:
#     bash scripts/run_sweep.sh                                  # the full paper grid (45 groups)
#     LAMBDAS="0.5" ALPHAS="0.2 0.8" bash scripts/run_sweep.sh   # a subset
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LAMBDAS=${LAMBDAS:-"0.1 0.3 0.5 0.7 0.9"}
ALPHAS=${ALPHAS:-"0 0.2 0.5 0.8 1 1.5 2 5 10"}
SUMMARY=${SUMMARY:-runs/sweep_summary.tsv}

n_total=$(( $(wc -w <<< "$LAMBDAS") * $(wc -w <<< "$ALPHAS") ))
echo "=================================================================="
echo " sweep: $n_total groups (λ: $LAMBDAS | α: $ALPHAS)"
echo " summary: $SUMMARY"
echo "=================================================================="

mkdir -p runs
[[ -f "$SUMMARY" ]] || printf "lambda\talpha\ttop1\ttop3\tagree\trun_dir\n" > "$SUMMARY"

i=0
for lam in $LAMBDAS; do
  for alpha in $ALPHAS; do
    i=$((i + 1))
    data_dir="data/clustered/train_lambda_${lam}"
    run_dir="runs/lambda_${lam}_alpha_${alpha}"

    if [[ ! -d "$data_dir" ]]; then
      echo "==== [$i/$n_total] λ=$lam α=$alpha  skipped: $data_dir not found"
      echo "     prepare it first: LAMBDAS=\"$lam\" bash scripts/prepare_data.sh"
      continue
    fi

    if [[ -f "$run_dir/best_model.pth" ]]; then
      echo "==== [$i/$n_total] λ=$lam α=$alpha  already exists, skipping training"
    else
      echo
      echo "==== [$i/$n_total] λ=$lam α=$alpha  starting training ===="
      LAMBDA="$lam" ALPHA="$alpha" RUN_DIR="$run_dir" bash scripts/train.sh
    fi

    RUN_DIR="$run_dir" bash scripts/eval.sh >/dev/null 2>&1 || {
      echo "  [warn] evaluation failed: $run_dir"; continue
    }

    top1=$(grep -oP 'File-avg Top-1:\s*\K[0-9.]+' "$run_dir/eval.log" | head -1)
    top3=$(grep -oP 'File-avg Top-3:\s*\K[0-9.]+' "$run_dir/eval.log" | head -1)
    agr=$(grep -oP 'File-avg Top3-overlap:\s*\K[0-9.]+' "$run_dir/eval.log" | head -1)
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$lam" "$alpha" "${top1:--}" "${top3:--}" "${agr:--}" "$run_dir" >> "$SUMMARY"
    echo "   -> Top-1 $top1 | Top-3 $top3 | Agree $agr"
  done
done

echo
echo "=================================================================="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "=================================================================="
