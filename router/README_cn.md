# Router训练

[English](README.md) | 简体中文

本目录是论文中 router 训练与评测部分的实现。

```
router/
├── src/          训练、聚类、评测脚本
├── scripts/      一键运行脚本
└── data/         训练数据
```

## 环境配置

```bash
conda create -n router python=3.10 -y
conda activate router

# 镜像设置
export HF_ENDPOINT=https://hf-mirror.com

pip install -r requirements.txt
```

依赖约 5.5 GB，核心是 torch 与 transformers 。backbone 用 `microsoft/mdeberta-v3-base`，
首次运行自动下载；如果机器离线可以把它拷到本地，加 `BACKBONE=/path/to/mdeberta-v3-base` 并设
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

装完运行这一条确认环境可用：

```bash
python -c "import torch,transformers;print(torch.cuda.is_available())"
```

建议使用显存16 GB及以上的显卡。

## 代码运行

训练数据已经随仓库提供（`data/clustered/train_lambda_0.5`，λ=0.5），可以直接开始训练：

```bash
# 论文默认设置：λ=0.5, α=0.2
bash scripts/train.sh

# 换一组参数
LAMBDA=0.3 ALPHA=0.8 bash scripts/train.sh

# 冒烟测试
TRAINING_STEPS=20 EVAL_STEPS=10 bash scripts/train.sh
```

训练产物在 `runs/lambda_<λ>_alpha_<α>/`，包含：`best_model.pth`、`model_order.json`、`config.txt`、
`training_log.json` 和准确率曲线。

四个脚本的分工：

- `scripts/prepare_data.sh`：数据链路（merge + 聚类）
- `scripts/train.sh`：训练单组
- `scripts/eval.sh`：评测单组
- `scripts/run_sweep.sh`：大量参数运行（论文网格是 5 个 λ × 9 个 α = 45 组），结果汇总到 `runs/sweep_summary.tsv`。

## 评测

```bash
bash scripts/eval.sh                                        # 评测默认那一组
RUN_DIR=runs/lambda_0.3_alpha_0.8 bash scripts/eval.sh      # 评测指定的一组
```

### 评测指标含义

真值是 `data/int_score/test`（每模型 0/1），`threshold=1.0` 视为答对。统计只在 eligible 题目上做平均，
即"该题至少有一个模型能答对"，没有任何模型能答对的题不计入。

- **Top-1-Hit**：打分最高的那个模型能答对；
- **Top-3-Hit**：打分前 3 名里至少有一个能答对；
- **Top-3-Agree**：预测的 top3 与真实 top3 的重合程度，不重合 0 个记 1、1 个记 0.6、2 个记 0.3、3 个记 0。

### 模型权重评测结果

这里涉及两套分值结构，先说明区别：

- **单分值**：每个模型 1 个 embedding，输出 N 维，即论文 Eq.(5) 的结构，也是 `src/train_onescore.py` 训练出来的结构，论文正文的三个指标由它得到。
- **双分值**：每个模型 2 个 embedding，输出 2N 维。我们发布的部署权重 `router_small.pth` 属于这一种，推理时只取第一个分值
  （`emoa/src/emoa/core/newemoa_v2.py` 中的 `probs[::2]`），实际只用到其中 N 个。

最初设计双输出，是希望第二组分数能承担额外的功能；实验中没有观察到收益，因此最终开源的训练代码是单分值版本，
部署权重沿用原结构、推理时只用第一组。

双分值部署权重在测试集上的实测结果，与论文的单分值对比如下：

| 指标 | 双分值（部署权重） | 单分值（论文） |
|---|---|---|
| Top-1-Hit | 0.9000 | 0.9073 |
| Top-3-Hit | 0.9822 | 0.9794 |
| Top-3-Agree | 0.9372 | 0.9615 |

两者差距很小、量级一致：单分值在 Top-1 与 Top-3-Agree 上略高，双分值的 Top-3-Hit 略高。

复现双分值（部署权重，`ckpt/` 指仓库根目录下存放权重的位置）：

```bash
python src/eval_deployed_twoscore.py \
    --trained_router_path ../ckpt/router_small.pth \
    --backbone_path microsoft/mdeberta-v3-base \
    --test_data_dir data/int_score/test \
    --score_index 0 --order sorted
```

复现单分值（论文口径，默认 λ=0.5、α=0.2）：

```bash
MODEL_ORDER=json bash scripts/train.sh
bash scripts/eval.sh
```

单次训练有随机性，数值会小幅浮动；完整的 λ×α 网格用 `scripts/run_sweep.sh` 复现。

## 数据来源

题目与标准答案取自公开数据集：MATH、ARC-c、MBPP、RACE-high、MMLU-biomed、GSM8k。
模型池里 5 个模型（Bio-Medical-Llama-3-8B、Qwen2.5-Coder-7B、
Qwen2.5-Math-7B、gemma-2-9b-it、Ministral-8B）对于问题各答一遍。

标签由两部分组成：

- `data/int_score/`：把每个模型的回答与标准答案比对得到 0/1。
- `data/reward_score/`：用 **InternLM2-1.8B-Reward** 对 (题目, 标准答案, 回答) 打分。

两部分按 λ 混合成训练标签。


## 如何应用到新的数据集上

如果你想要对你自己的数据集进行打分，可以参考以下步骤：

**第一步：收集回答。** 每条 query 让你池子里每个模型各答一遍，推理方式不限，每条 query 写一行 jsonl：

```json
{"question": "题目文本，也是后面 join 的键",
 "prompt": "实际喂给模型的完整文本",
 "responses": {"model_a": "...", "model_b": "..."}}
```


**第二步：生成 0/1。** 把每个回答与标准答案比对，得到：

```json
[{"question": "...", "scores": {"model_a": 1, "model_b": 0}}]
```



**第三步：生成 reward 分。** 可以参考我们提供的 `src/reward_score.py`：

```bash
python src/reward_score.py \
    --input responses.jsonl \
    --out   data/reward_score/train/your_dataset.json \
    --judger internlm/internlm2-1_8b-reward
```


**第四步：混合、聚类、训练。**

```bash
LAMBDAS="0.5" bash scripts/prepare_data.sh
bash scripts/train.sh
```

流程和用我们提供的数据完全一样。几个注意点：

- `prepare_data.sh` 会跳过已经存在的产物；换了题目集合想强制重建，加 `FORCE=1`；
- 只换 λ 时聚类结果会被继承，不用重复计算；
- `--model_order` 要前后保持一致，训练脚本会把对应关系写进 `model_order.json`，评测时自动读取。
