"""Usage:
    python src/eval_deployed_twoscore.py \
        --trained_router_path ../ckpt/router_small.pth \
        --backbone_path microsoft/mdeberta-v3-base \
        --test_data_dir data/int_score/test \
        --score_index 0
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Model

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_onescore import RouterModule  # noqa: E402

PROB_ACTIVATIONS = ("softmax", "sigmoid")
TOP3_SCORE_MAP = {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.0}


class ScoreDataset(Dataset):
    def __init__(self, data_path: str, models: List[str], tokenizer, max_length: int = 512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.models = models
        self.tokenizer = tokenizer
        self.max_length = max_length
        for rec in self.data[:100]:
            missing = [m for m in models if m not in rec["scores"]]
            assert not missing, f"{data_path}: gt is missing models {missing}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        enc = self.tokenizer(
            rec["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        gt = torch.tensor([float(rec["scores"][m]) for m in self.models], dtype=torch.float32)
        return enc["input_ids"].flatten(), enc["attention_mask"].flatten(), gt


def evaluate_file(model, data_path, models, tokenizer, device, batch_size,
                  threshold, prob_activation, score_index, top_k=3):
    ds = ScoreDataset(data_path, models, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    num_models = len(models)
    k = min(top_k, num_models)

    top1_hits = topk_hits = eligible = total = 0
    overlap_sum_all = overlap_sum_elig = 0.0

    with torch.no_grad():
        for input_ids, attn, gt in dl:
            input_ids, attn = input_ids.to(device), attn.to(device)
            logits, _ = model(t=1, input_ids=input_ids, attention_mask=attn)
            probs = torch.sigmoid(logits) if prob_activation == "sigmoid" else torch.softmax(logits, dim=-1)

            if probs.shape[1] == 2 * num_models:
                pred = probs[:, score_index::2]
            elif probs.shape[1] == num_models:
                pred = probs
            else:
                raise ValueError(f"output dimension {probs.shape[1]} does not match the number of models {num_models}")

            _, pred_top = torch.topk(pred, k=k, dim=1)
            _, pred_top3 = torch.topk(pred, k=k, dim=1)

            for r in range(pred.shape[0]):
                g = gt[r]
                total += 1
                has_truth = bool((g >= threshold).any().item())

                if has_truth:
                    eligible += 1
                    top1_hits += int(float(g[pred_top[r][0]]) >= threshold)
                    topk_hits += int(any(float(g[i]) >= threshold for i in pred_top[r].tolist()))

                truth_vals, _ = torch.topk(g, k=k)
                truth_set = (g >= truth_vals[-1]).nonzero(as_tuple=True)[0].tolist()
                mismatch = sum(1 for i in pred_top3[r].tolist() if i not in truth_set)
                s = TOP3_SCORE_MAP[min(mismatch, k)]
                overlap_sum_all += s
                if has_truth:
                    overlap_sum_elig += s

    return {
        "file": os.path.basename(data_path),
        "questions": total,
        "eligible": eligible,
        "top1": (top1_hits / eligible) if eligible else 0.0,
        "topk": (topk_hits / eligible) if eligible else 0.0,
        "overlap_all": (overlap_sum_all / total) if total else 0.0,
        "overlap_elig": (overlap_sum_elig / eligible) if eligible else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trained_router_path", required=True)
    p.add_argument("--backbone_path", required=True)
    p.add_argument("--test_data_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--prob_activation", choices=PROB_ACTIVATIONS, default="softmax")
    p.add_argument("--score_index", type=int, default=0, choices=[0, 1],
                   help="0=probs[::2] (what deployment actually uses), 1=probs[1::2] (control)")
    p.add_argument("--order", default="sorted", choices=["sorted", "json"],
                   help="model order used when trained_router was trained: twoscore->sorted, onescore->json (original key order)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, truncation_side="left", padding=True)
    backbone = DebertaV2Model.from_pretrained(args.backbone_path)

    state_dict = torch.load(args.trained_router_path, map_location="cpu", weights_only=True)
    node_size = state_dict["embeddings.weight"].shape[0]
    model = RouterModule(backbone, hidden_state_dim=backbone.config.hidden_size,
                         node_size=node_size, similarity_function="cos").to(device)
    model.load_state_dict(state_dict)
    model.eval()

    files = sorted(f for f in os.listdir(args.test_data_dir) if f.endswith(".json"))
    with open(os.path.join(args.test_data_dir, files[0]), "r", encoding="utf-8") as f:
        first = json.load(f)
    # twoscore: sorted keys; onescore: json order
    models = sorted(first[0]["scores"].keys()) if args.order == "sorted" else list(first[0]["scores"].keys())
    num_models = len(models)

    print(f"checkpoint node_size = {node_size} (models = {num_models}, expect {2 * num_models} for dual-score)")
    print(f"model order ({args.order})  = {models}")
    print(f"score_index           = {args.score_index}  ({'probs[::2] deployment usage' if args.score_index == 0 else 'probs[1::2] control'})")
    print(f"threshold={args.threshold}  activation={args.prob_activation}  device={device}\n")

    rows = [evaluate_file(model, os.path.join(args.test_data_dir, f), models, tokenizer,
                          device, args.batch_size, args.threshold, args.prob_activation,
                          args.score_index, args.top_k)
            for f in files]

    hdr = f"{'File':<24}{'Questions':>10}{'Eligible-Q':>11}{'Top1-Hits':>10}{'TopK-Hits':>10}{'Top1-Score':>11}{'TopK-Score':>11}{'Top3-Ovlp':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:<24}{r['questions']:>10}{r['eligible']:>11}"
              f"{int(round(r['top1'] * r['eligible'])):>10}{int(round(r['topk'] * r['eligible'])):>10}"
              f"{r['top1']:>11.6f}{r['topk']:>11.6f}{r['overlap_all']:>10.6f}")

    n = len(rows)
    file_avg_top1 = sum(r["top1"] for r in rows) / n
    file_avg_topk = sum(r["topk"] for r in rows) / n
    file_avg_ovlp = sum(r["overlap_all"] for r in rows) / n
    elig_avg_ovlp = sum(r["overlap_elig"] for r in rows) / n

    tot_q = sum(r["questions"] for r in rows)
    tot_elig = sum(r["eligible"] for r in rows)
    print("\n========== Overall ==========")
    print(f"Total files: {n}   Total questions: {tot_q}   Total eligible: {tot_elig}")
    print(f"File-avg Top-1        : {file_avg_top1:.6f}")
    print(f"File-avg Top-{args.top_k}        : {file_avg_topk:.6f}")
    print(f"File-avg Top3-overlap : {file_avg_ovlp:.6f}   (denominator = all questions)")
    print(f"File-avg Top3-overlap : {elig_avg_ovlp:.6f}   (denominator = eligible questions, for reference)")


if __name__ == "__main__":
    main()
