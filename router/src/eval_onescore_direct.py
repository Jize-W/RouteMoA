"""Usage:
    python src/eval_onescore_direct.py \
        --trained_router_path runs/lambda_0.5_alpha_0.2/best_model.pth \
        --backbone_path microsoft/mdeberta-v3-base \
        --test_data_dir data/int_score/test
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Model
from pathlib import Path

from train_onescore import RouterDataset, RouterModule


def collect_json_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".json")
    )


def get_best_cuda_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    best, max_free = "cuda:0", 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > max_free:
            best, max_free = f"cuda:{i}", free
    return best


def evaluate_file(
    router_model: RouterModule,
    data_path: str,
    tokenizer,
    batch_size: int,
    device: str,
    top_k: int,
    threshold: float,
    prob_activation: str,
    model_order: str = "json",
) -> Dict[str, float]:
    dataset = RouterDataset(data_path=data_path, model_order=model_order)
    dataset.register_tokenizer(tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_names = dataset.router_node
    num_models = len(model_names)
    k = min(top_k, num_models)
    top3_k = min(3, num_models)

    top1_hits = 0
    topk_hits = 0
    total = 0
    total_hits_den = 0
    top3_overlap_sum = 0.0
    top3_perfect_matches = 0
    top3_tie_expanded_cases = 0

    top3_score_map = {
        0: 1.0,
        1: 0.6,
        2: 0.3,
        3: 0.0,
    }

    with torch.no_grad():
        for batch in data_loader:
            inputs, scores, _, _ = batch
            inputs = inputs.to(device)
            scores = scores.to(device)

            logits, _ = router_model.forward(**inputs)
            if prob_activation == "sigmoid":
                probs = torch.sigmoid(logits)
            else:
                probs = nn.Softmax(dim=1)(logits)

            topk_values, topk_indices = torch.topk(probs, k=k, dim=1)
            _, top3_pred_indices = torch.topk(probs, k=top3_k, dim=1)

            for row_idx in range(probs.size(0)):
                row_scores = scores[row_idx]
                top_indices = topk_indices[row_idx].tolist()
                top1_idx = top_indices[0]

                sample_has_truth = bool((row_scores >= threshold).any().item())

                if sample_has_truth:
                    top1_hit = float(row_scores[top1_idx].item()) >= threshold
                    topk_hit = any(float(row_scores[idx].item()) >= threshold for idx in top_indices)
                    top1_hits += int(top1_hit)
                    topk_hits += int(topk_hit)
                    total_hits_den += 1

                total += 1

                truth_top_values, _ = torch.topk(row_scores, k=top3_k)
                truth_cutoff = truth_top_values[-1].item()
                truth_top_set = (row_scores >= truth_cutoff).nonzero(as_tuple=True)[0].tolist()

                pred_top_indices = top3_pred_indices[row_idx].tolist()
                mismatch_count = sum(1 for idx in pred_top_indices if idx not in truth_top_set)
                mismatch_key = min(mismatch_count, 3)
                overlap_score = top3_score_map.get(mismatch_key, 0.0)

                top3_overlap_sum += overlap_score
                if mismatch_count == 0:
                    top3_perfect_matches += 1
                if len(truth_top_set) > top3_k:
                    top3_tie_expanded_cases += 1

    top1_score = (top1_hits / total_hits_den) if total_hits_den else 0.0
    topk_score = (topk_hits / total_hits_den) if total_hits_den else 0.0
    top3_overlap_avg = (top3_overlap_sum / total) if total else 0.0

    return {
        "file": data_path,
        "total": total,
        "total_hits": total_hits_den,
        "top1_hits": top1_hits,
        "topk_hits": topk_hits,
        "top1_score": top1_score,
        "topk_score": topk_score,
        "k": k,
        "threshold": threshold,
        "top3_overlap_sum": top3_overlap_sum,
        "top3_overlap_avg": top3_overlap_avg,
        "top3_perfect_matches": top3_perfect_matches,
        "top3_tie_expanded_cases": top3_tie_expanded_cases,
    }





def evaluate(
    router_model: RouterModule,
    dataset_paths: List[str],
    tokenizer,
    batch_size: int,
    device: str,
    top_k: int,
    threshold: float,
    prob_activation: str,
    model_order: str = "json",
):
    per_file = []
    total_top1 = 0
    total_topk = 0
    total_q = 0
    total_q_hits = 0
    sum_top1_score = 0.0
    sum_topk_score = 0.0
    total_top3_overlap_sum = 0.0
    sum_top3_overlap_avg = 0.0
    total_top3_perfect_matches = 0
    total_top3_tie_expanded_cases = 0

    for path in dataset_paths:
        res = evaluate_file(
            router_model,
            path,
            tokenizer,
            batch_size,
            device,
            top_k,
            threshold,
            prob_activation,
            model_order,
        )
        per_file.append(res)
        total_top1 += res["top1_hits"]
        total_topk += res["topk_hits"]
        total_q += res["total"]
        total_q_hits += res["total_hits"]
        sum_top1_score += res["top1_score"]
        sum_topk_score += res["topk_score"]
        total_top3_overlap_sum += res["top3_overlap_sum"]
        sum_top3_overlap_avg += res["top3_overlap_avg"]
        total_top3_perfect_matches += res["top3_perfect_matches"]
        total_top3_tie_expanded_cases += res["top3_tie_expanded_cases"]

    overall = {
        "total_files": len(per_file),
        "total_questions": total_q,
        "total_top1_hits": total_top1,
        "total_topk_hits": total_topk,
        "file_top1_avg": (sum_top1_score / len(per_file)) if per_file else 0.0,
        "file_topk_avg": (sum_topk_score / len(per_file)) if per_file else 0.0,
        "weighted_top1_avg": (total_top1 / total_q_hits) if total_q_hits else 0.0,
        "weighted_topk_avg": (total_topk / total_q_hits) if total_q_hits else 0.0,
        "file_top3_overlap_avg": (sum_top3_overlap_avg / len(per_file)) if per_file else 0.0,
        "weighted_top3_overlap_avg": (total_top3_overlap_sum / total_q) if total_q else 0.0,
        "total_top3_perfect_matches": total_top3_perfect_matches,
        "total_top3_tie_expanded_cases": total_top3_tie_expanded_cases,
    }
    return per_file, overall


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate router (one-score) directly without writing prediction files",
    )
    parser.add_argument("--test_data_paths", nargs="+", default=None, help="List of test json files or a directory")
    parser.add_argument("--test_data_dir", type=str, default="data/int_score/test", help="If set, collect all json files under this directory")
    parser.add_argument("--trained_router_path", type=str, default="runs/default/best_model.pth", help="Path to trained router state_dict")
    parser.add_argument("--backbone_path", type=str, default="microsoft/mdeberta-v3-base", help="Backbone checkpoint path (HF repo id or a local directory)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=3, help="Top-K used for success metric")
    parser.add_argument("--threshold", type=float, default=1.0, help="Score threshold for success")
    parser.add_argument("--prob_activation", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--model_order", choices=["auto", "sorted", "json"], default="auto",
                        help="model order: auto=read model_order.json next to the checkpoint (falls back to json for old checkpoints)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import time

    time_start = time.time()

    if args.test_data_dir:
        dataset_paths = collect_json_files(args.test_data_dir)
    else:
        dataset_paths = args.test_data_paths or []
        if len(dataset_paths) == 1 and os.path.isdir(dataset_paths[0]):
            dataset_paths = collect_json_files(dataset_paths[0])
    if not dataset_paths:
        raise ValueError("No test data files provided")

    device = get_best_cuda_device()
    print(f"Selected device: {device}")
    print("Test files:", dataset_paths)

    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone_path,
        truncation_side="left",
        padding=True,
    )
    encoder_model = DebertaV2Model.from_pretrained(args.backbone_path)

    state_dict = torch.load(args.trained_router_path, map_location=device)
    node_size = state_dict["embeddings.weight"].shape[0]
    router_model = RouterModule(
        encoder_model,
        hidden_state_dim=768,
        node_size=node_size,
        similarity_function="cos",
    ).to(device)
    router_model.load_state_dict(state_dict)
    router_model.eval()

    # auto: read model_order.json, fallback "json"
    model_order = args.model_order
    if model_order == "auto":
        mo_path = os.path.join(os.path.dirname(os.path.abspath(args.trained_router_path)), "model_order.json")
        if os.path.isfile(mo_path):
            with open(mo_path, "r", encoding="utf-8") as f:
                mo = json.load(f)
            model_order = mo.get("order_type", "json")
            print(f"[model_order] auto -> {model_order}  ({mo_path})")
            print(f"[model_order] index_map = {mo.get('index_map')}")
        else:
            model_order = "json"
            print("[model_order] auto -> json (model_order.json not found, treated as an old checkpoint)")

    per_file, overall = evaluate(
        router_model=router_model,
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device,
        top_k=args.top_k,
        threshold=args.threshold,
        prob_activation=args.prob_activation,
        model_order=model_order,
    )

    print("\n========== Per-file Scores ==========")
    header = (
        f"{'File':<40}{'Questions':>12}{'Eligible-Q':>12}{'Top1-Hits':>12}{'TopK-Hits':>12}"
        f"{'Top1-Score':>14}{'TopK-Score':>14}{'Top3-Ovlp':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in per_file:
        print(
            f"{os.path.basename(r['file']):<40}"
            f"{r['total']:>12d}"
            f"{r['total_hits']:>12d}"
            f"{r['top1_hits']:>12d}"
            f"{r['topk_hits']:>12d}"
            f"{r['top1_score']:>14.6f}"
            f"{r['topk_score']:>14.6f}"
            f"{r['top3_overlap_avg']:>12.6f}"
        )

    print("\n========== Overall (Weighted) ==========")
    print(f"Total files: {overall['total_files']}")
    print(f"Total questions: {overall['total_questions']}")
    print(f"Total Top-1 hits: {overall['total_top1_hits']}")
    print(f"Total Top-{args.top_k} hits: {overall['total_topk_hits']}")
    print(f"File-avg Top-1: {overall['file_top1_avg']:.6f} (each file equal weight)")
    print(f"File-avg Top-{args.top_k}: {overall['file_topk_avg']:.6f} (each file equal weight)")
    print(f"File-avg Top3-overlap: {overall['file_top3_overlap_avg']:.6f} (each file equal weight)")

    print(f"Weighted Top3-overlap (by questions): {overall['weighted_top3_overlap_avg']:.6f}")


    time_end = time.time()
    print(f"\nTotal evaluation time: {time_end - time_start:.2f} seconds")
if __name__ == "__main__":
    main(sys.argv[1:])
