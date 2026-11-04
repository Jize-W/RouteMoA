# Router Training

English | [简体中文](README_cn.md)

This directory is the implementation of the router training and evaluation part of the paper.

```
router/
├── src/          training, clustering, evaluation scripts
├── scripts/      one-command scripts
└── data/         training data
```

## Environment Setup

```bash
conda create -n router python=3.10 -y
conda activate router

# mirror
export HF_ENDPOINT=https://hf-mirror.com

pip install -r requirements.txt
```

The dependencies take about 5.5 GB, the core ones being torch and transformers. The backbone is
`microsoft/mdeberta-v3-base`, downloaded automatically on the first run; if the machine is offline,
you can copy it locally, add `BACKBONE=/path/to/mdeberta-v3-base` and set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

After installation, run this line to confirm that the environment is usable:

```bash
python -c "import torch,transformers;print(torch.cuda.is_available())"
```

We recommend using a GPU with 16 GB of memory or more.

## Running the Code

The training data is already provided with the repository (`data/clustered/train_lambda_0.5`, λ=0.5),
so training can start directly:

```bash
# paper default: λ=0.5, α=0.2
bash scripts/train.sh

# another parameter set
LAMBDA=0.3 ALPHA=0.8 bash scripts/train.sh

# smoke test
TRAINING_STEPS=20 EVAL_STEPS=10 bash scripts/train.sh
```

Training artifacts are written to `runs/lambda_<λ>_alpha_<α>/`, including: `best_model.pth`,
`model_order.json`, `config.txt`, `training_log.json` and the accuracy curve.

Responsibilities of the four scripts:

- `scripts/prepare_data.sh`: the data pipeline (merge + clustering)
- `scripts/train.sh`: train a single group
- `scripts/eval.sh`: evaluate a single group
- `scripts/run_sweep.sh`: run a large number of parameter combinations (the paper grid is 5 λ × 9 α = 45 groups),
  with the results collected into `runs/sweep_summary.tsv`.

## Evaluation

```bash
bash scripts/eval.sh                                        # evaluate the default group
RUN_DIR=runs/lambda_0.3_alpha_0.8 bash scripts/eval.sh      # evaluate a specific group
```

### Meaning of the Evaluation Metrics

The ground truth is `data/int_score/test` (0/1 per model), and `threshold=1.0` is treated as correct.
The statistics are averaged only over eligible questions, i.e. "at least one model answers this question correctly";
questions that no model answers correctly are not counted.

- **Top-1-Hit**: the highest-scoring model answers correctly;
- **Top-3-Hit**: at least one of the top-3 scoring models answers correctly;
- **Top-3-Agree**: the degree of overlap between the predicted top3 and the true top3, where 0 mismatches
  score 1, 1 scores 0.6, 2 scores 0.3, and 3 scores 0.

### Evaluation Results of the Model Weights

Two score structures are involved here, so their difference is explained first:

- **Single-score**: each model has 1 embedding and the output is N-dimensional, i.e. the structure of Eq.(5)
  in the paper, and also the structure trained by `src/train_onescore.py`; the three metrics in the main text
  of the paper come from it.
- **Dual-score**: each model has 2 embeddings and the output is 2N-dimensional. The deployment weights
  `router_small.pth` we release belong to this kind; at inference only the first score is taken
  (`probs[::2]` in `emoa/src/emoa/core/newemoa_v2.py`), so only N of them are actually used.

The dual output was originally designed in the hope that the second group of scores could take on an
additional function; no benefit was observed in the experiments, so the training code finally open-sourced
is the single-score version, while the deployment weights keep the original structure and use only the
first group at inference.

The measured results of the dual-score deployment weights on the test set, compared with the single-score
numbers of the paper, are as follows:

| Metric | Dual-score (deployment weights) | Single-score (paper) |
|---|---|---|
| Top-1-Hit | 0.9000 | 0.9073 |
| Top-3-Hit | 0.9822 | 0.9794 |
| Top-3-Agree | 0.9372 | 0.9615 |

The two are very close and of the same magnitude: the single-score one is slightly higher on Top-1 and
Top-3-Agree, while the dual-score one is slightly higher on Top-3-Hit.

Reproduce the dual-score result (deployment weights; `ckpt/` refers to the location of the weights
downloaded under the repository root):

```bash
python src/eval_deployed_twoscore.py \
    --trained_router_path ../ckpt/router_small.pth \
    --backbone_path microsoft/mdeberta-v3-base \
    --test_data_dir data/int_score/test \
    --score_index 0 --order sorted
```

Reproduce the single-score result (the paper setting, default λ=0.5, α=0.2):

```bash
MODEL_ORDER=json bash scripts/train.sh
bash scripts/eval.sh
```

`MODEL_ORDER=json` is the setting behind the table in the paper; training writes the order into
`model_order.json` and evaluation reads it automatically, so it does not need to be specified again.

A single training run is stochastic and the numbers may fluctuate slightly; the full λ×α grid is reproduced
with `scripts/run_sweep.sh`.

## Data Source

The questions and ground-truth answers are taken from public datasets: MATH, ARC-c, MBPP, RACE-high,
MMLU-biomed, GSM8k.
The 5 models in the pool (Bio-Medical-Llama-3-8B, Qwen2.5-Coder-7B, Qwen2.5-Math-7B, gemma-2-9b-it,
Ministral-8B) each answer every question once.

The labels consist of two parts:

- `data/int_score/`: compare each model's answer with the ground truth to obtain 0/1.
- `data/reward_score/`: use **InternLM2-1.8B-Reward** to score (question, ground truth, answer).

The two parts are mixed by λ into the training labels.

## How to Apply to a New Dataset

If you want to score your own dataset, you can refer to the following steps.

**Step 1: collect the answers.** For each query, let every model in your pool answer once; the inference
setup is not restricted; write one line of jsonl per query:

```json
{"question": "the question text, also the join key later",
 "prompt": "the full text actually fed to the model",
 "responses": {"model_a": "...", "model_b": "..."}}
```

**Step 2: generate the 0/1 labels.** Compare each answer with the ground truth to obtain:

```json
[{"question": "...", "scores": {"model_a": 1, "model_b": 0}}]
```

**Step 3: generate the reward scores.** You can refer to `src/reward_score.py` provided by us:

```bash
python src/reward_score.py \
    --input responses.jsonl \
    --out   data/reward_score/train/your_dataset.json \
    --judger internlm/internlm2-1_8b-reward
```

**Step 4: mix, cluster and train.**

```bash
LAMBDAS="0.5" bash scripts/prepare_data.sh
bash scripts/train.sh
```

The flow is exactly the same as using the data we provide. A few things to note:

- `prepare_data.sh` skips outputs that already exist; to force a rebuild after changing the question set,
  add `FORCE=1`;
- when only λ changes, the clustering result is inherited, so there is no need to recompute it;
- `--model_order` must stay consistent; the training script writes the mapping into `model_order.json`,
  and evaluation reads it automatically.
