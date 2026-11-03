# RouteMoA

English | [简体中文](README_cn.md)

**[ACL 2026] RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents**

Welcome to the official repository for **RouteMoA** ([Paper link](https://arxiv.org/abs/2601.18130)). This project introduces an efficient and dynamic routing mechanism designed to boost the performance of Mixture-of-Agents (MoA) architectures without relying on pre-inference steps.

<p align="center">
  <img src="asset/compare.png" alt="Concept comparison between our RouteMoA and previous MoA-based methods." width="100%">
  <br>
  <em>Concept comparison between our RouteMoA and previous MoA-based methods.</em>
</p>

---

## Table of Contents

- [RouteMoA](#routemoa)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start — Small Model Pool](#quick-start--small-model-pool)
    - [1. Deploy the LLMs](#1-deploy-the-llms)
    - [2. Start the RouteMoA Service](#2-start-the-routemoa-service)
    - [3. Evaluate the Service](#3-evaluate-the-service)
  - [Quick Start — Large Model Pool](#quick-start--large-model-pool)
    - [1. Configure API Access](#1-configure-api-access)
    - [2. Start the Services](#2-start-the-services)
    - [3. Evaluate the Service](#3-evaluate-the-service-1)
  - [Router Training](#router-training)
  - [Acknowledgements](#acknowledgements)

---

## Installation

We highly recommend using `conda` to create an isolated environment.

> **System Requirement:** We strongly recommend using a Linux (x86_64) operating system. This guide is based on Ubuntu 22.04, and other operating systems are currently not officially supported.

```bash
conda create -n YOUR_ENV_NAME python=3.10 -y
conda activate YOUR_ENV_NAME

# Install the OpenCompass evaluation framework (for small model pool experiments)
cd opencompass
pip install -e .

# Install the EMoA core module (small model pool)
cd ../emoa
pip install -e .

# Install the EMoA large model pool module
cd ../emoa_large
pip install -e .
```

---

## Quick Start — Small Model Pool

### 1. Deploy the LLMs

We suggest downloading the LLM checkpoints locally first. Our experiments were conducted using five NVIDIA A800 80GB GPUs. For reproducibility, **We strongly recommend deploying each model on a dedicated high-performance NVIDIA GPU with more than 70GB of VRAM.**

Set up your local checkpoint directory:
```bash
mkdir -p </path/to/your/local/checkpoint/folder>
```

Install the HuggingFace CLI tool:
```bash
pip install -U huggingface_hub
```

Download the required models from HuggingFace:
```bash
# Download Bio-Medical Llama
huggingface-cli download ContactDoctor/Bio-Medical-Llama-3-8B \
  --token <your-huggingface-token> \
  --local-dir </path/to/your/local/checkpoint/folder>/Bio-Medical-Llama-3-8B

# Download Qwen models
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir </path/to/your/local/checkpoint/folder>/Qwen2.5-Coder-7B-Instruct

huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
  --local-dir </path/to/your/local/checkpoint/folder>/Qwen2.5-Math-7B-Instruct

# Download Gemma and Ministral models
huggingface-cli download google/gemma-2-9b-it \
  --token <your-huggingface-token> \
  --local-dir </path/to/your/local/checkpoint/folder>/gemma-2-9b-it

huggingface-cli download mistralai/Ministral-8B-Instruct-2410 \
  --local-dir </path/to/your/local/checkpoint/folder>/Ministral-8B-Instruct-2410
```

Once the models are ready, deploy them using LMDeploy:
```bash
bash lmdeploy.sh
```

### 2. Start the RouteMoA Service

Next, set up the router and launch the core services.

1. **Download the router weights** from either of these links:
   - [GitHub Release](https://github.com/Jize-W/RouteMoA/releases/download/weights-v1/router_small.pth)
   - [Google Drive](https://drive.google.com/file/d/1cMntfcUJ6mKf5a2bsgwwWo4ehCLWqAuw/view?usp=sharing)
2. Place the downloaded weights into your local `checkpoints/` directory.
3. **Download the router backbone** (`mdeberta-v3-base`) from the official Microsoft project:
   - Model weights: [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)
   - Or simply use the CLI:
   ```bash
   huggingface-cli download microsoft/mdeberta-v3-base --local-dir </path/to/mdeberta-v3-base>
   ```
4. **Update the config file** (`emoa/configs/emoa_v2.json`):
   - Set `router_pth_path` to the absolute path of your downloaded router weights.
   - Set `router_backbone` to the absolute path of your `mdeberta-v3-base` folder.

Now, start the services:
```bash
conda activate YOUR_ENV_NAME

# Start the EMoA service
python3 -m emoa.serve.app_v2 --config emoa/configs/emoa_v2.json --host 0.0.0.0 --port 10666

# Start the SMoA service
python3 -m emoa.serve.smoa --config emoa/configs/smoa.json --host 0.0.0.0 --port 10667
```

### 3. Evaluate the Service

We use OpenCompass to evaluate the performance of our EMoA service.

First, run the inference:
```bash
conda activate YOUR_ENV_NAME
opencompass examples/eval_emoa.py -r latest --mode infer --dump-eval-details
```

After inference completes, run the evaluation to calculate the scores:
```bash
opencompass examples/eval_emoa.py -r latest --mode eval --dump-eval-details
```

If you need to analyze the API costs and latency, we provide a handy script:
```bash
cd opencompass
python bill_stat.py
```

> **Note on Reproducibility:** To demonstrate the reproducibility of our experiments, we have provided the full OpenCompass evaluation results. You can download and verify them [here](https://drive.google.com/file/d/1QAIy1lxqvXrPFoQj0g6tjHXMWxowH0Je/view?usp=sharing).

---

## Quick Start — Large Model Pool

The large model pool experiment calls external LLM APIs (e.g., DeepSeek, Qwen) instead of deploying local models. All code is in the `emoa_large/` directory.

### 1. Configure API Access

**a) Install dependencies:**
```bash
cd emoa_large
pip install -e .
```

**b) Fill in your API credentials** in `emoa_large/configs/api_info.csv`:

Open the file and replace `YOUR_API_BASE_URL` and `YOUR_API_KEY` for each model with your actual API endpoint and key. The file uses the standard OpenAI-compatible API format:

```
model,model_name,model_id,input_price,output_price,api_type,api_base,api_key
deepseek-ai/DeepSeek-V3-0324,deepseek-v3-0324,,0.28,1.14,openai,https://api.deepseek.com/v1,YOUR_KEY
...
```

You do not need to populate all 15 models — only configure the models you plan to use and update the `candidate_models` list in the relevant config JSON accordingly.

**c) Download the router weights** (required for RouteMoA only):

Download `router_large.pth` from [GitHub Release](https://github.com/Jize-W/RouteMoA/releases/download/weights-v1/router_large.pth) or [Google Drive](https://drive.google.com/file/d/1ixpMzKtw52OIG3pJKviY19iW29BGJHpx/view?usp=drive_link), then place it at:
```
emoa_large/weights/router_large.pth
```

**d) Download the router backbone** (`microsoft/mdeberta-v3-base`):
```bash
huggingface-cli download microsoft/mdeberta-v3-base --local-dir /path/to/mdeberta-v3-base
```
Then update `router_backbone` in `emoa_large/configs/routemoa.json` to the local path, or leave it as `"microsoft/mdeberta-v3-base"` to download automatically from HuggingFace.

### 2. Start the Services

All three methods expose an OpenAI-compatible `/v1/chat/completions` endpoint.

**MoA** (baseline):
```bash
cd emoa_large
python3 -m emoa.serve.app_moa --config configs/moa.json --host 0.0.0.0 --port 10078
```

**RouteMoA** (ours):
```bash
cd emoa_large
python3 -m emoa.serve.app_routemoa --config configs/routemoa.json --host 0.0.0.0 --port 10079
```

**SMoA** (baseline):
```bash
cd emoa_large
python3 -m emoa.serve.app_smoa --config configs/smoa.json --host 0.0.0.0 --port 10080
```

You can verify a service is running by checking its health endpoint:
```bash
curl http://localhost:10078/health
```

### 3. Evaluate the Service

> **Note on evaluation methodology:** The large model pool results in Table 1 of the paper were produced using an internal evaluation platform that is not publicly available. We provide an equivalent standalone evaluation suite in `emoa_large/eval/` that reproduces the same benchmark, metrics, and results using only open-source tools.

The `emoa_large/eval/` directory contains everything needed to reproduce the paper's large model pool results.

**Step 1 — Install evaluation dependencies:**
```bash
pip install openai
pip install lawrouge          # for RougeMetric (lcsts)
pip install pkuseg nltk       # for GEC F1 (nlpcc2018_task2, conll2014)
```

For GEC F1, also install `m2scorer`:
```bash
git clone https://github.com/nusnlp/m2scorer
export PYTHONPATH=/path/to/m2scorer:$PYTHONPATH
```

**Step 2 — Run inference against a running service:**
```bash
cd emoa_large/eval
python inference.py \
    --base_url  http://localhost:10079/v1 \
    --api_key   dummy \
    --model     routemoa \
    --input     benchmark_questions.json \
    --output    predictions_routemoa.json \
    --workers   4
```

**Step 3 — Evaluate predictions:**
```bash
python evaluate.py \
    --predictions     predictions_routemoa.json \
    --benchmark       benchmark_questions.json \
    --output          eval_results_routemoa.json \
    --judge_base_url  https://api.openai.com/v1 \
    --judge_api_key   YOUR_OPENAI_KEY \
    --judge_model     gpt-4o
```

The script prints per-dataset scores, category averages, and a global average to stdout.

> **Pre-computed results** are available in `emoa_large/eval/` for all three methods (`moa.json`, `routemoa.json`, `smoa.json`) along with a cross-model summary in `summary_all.json`. See `emoa_large/eval/README.md` for the full benchmark description and paper results.

---

## Router Training

Our router training code is built on top of the [RouterDC](https://github.com/shuhao02/RouterDC) framework.

You can refer to the instructions in the RouterDC repository to train your own Router with your own data. Currently, we have already released the trained router weights, and you can also refer to our training method in this folder [Router](./router).

---

## Acknowledgements

This project is built upon the great work from the open-source community. We sincerely thank:
- [OpenCompass](https://github.com/open-compass/opencompass)
- [RouterDC](https://github.com/shuhao02/RouterDC)
