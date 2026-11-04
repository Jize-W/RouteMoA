"""Usage:
    python src/cluster_generate.py \
        --data_dir data/merged/train_lambda_0.5 \
        --out_dir  data/clustered/train_lambda_0.5 \
        --backbone microsoft/mdeberta-v3-base \
        --n_clusters 6
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Model

try:
    from openTSNE import TSNE
    _HAS_OPEN_TSNE = True
except ImportError:
    from sklearn.manifold import TSNE
    _HAS_OPEN_TSNE = False

from sklearn.cluster import KMeans

from train_onescore import RouterDataset, RouterModule


def collect_json_files(directory: str):
    if not os.path.isdir(directory):
        raise SystemExit(f"{directory} is not a valid directory")
    return sorted(
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if fn.lower().endswith(".json")
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="t-SNE + k-means clustering of training queries")
    p.add_argument("--data_dir", default="data/merged/train_lambda_0.5",
                   help="merged json directory (question + scores, no cluster_id yet)")
    p.add_argument("--out_dir", default="data/clustered/train_lambda_0.5",
                   help="output directory; each input json gets a cluster_id field added")
    p.add_argument("--backbone", default="microsoft/mdeberta-v3-base",
                   help="mDeBERTaV3-base encoder: HF repo id or local directory")
    p.add_argument("--n_clusters", type=int, default=6, help="Q, number of k-means clusters (paper: 6)")
    p.add_argument("--num_per_dataset", type=int, default=2000,
                   help="cap (and pad) the number of samples per dataset used for clustering, kept identical to the released data")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsne_jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="number of openTSNE threads")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    dataset_paths = collect_json_files(args.data_dir)
    print(f"[cluster] {len(dataset_paths)} json files:\n  " + "\n  ".join(dataset_paths))

    print(f"[cluster] backbone: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, truncation_side="left", padding=True)
    encoder_model = DebertaV2Model.from_pretrained(args.backbone).to(args.device).eval()

    router_datasets = [
        RouterDataset(path, data_type="multi_attempt", dataset_id=i, size=args.num_per_dataset)
        for i, path in enumerate(dataset_paths)
    ]
    for ds in router_datasets:
        ds.register_tokenizer(tokenizer)

    router_model = RouterModule(
        encoder_model,
        hidden_state_dim=768,
        node_size=len(router_datasets[0].router_node),
        similarity_function="cos",
    ).to(args.device).eval()

    model_names = router_datasets[0].router_node
    for ds in router_datasets[1:]:
        assert ds.router_node == model_names, (
            f"model order mismatch: {ds.router_node} vs {model_names}"
        )
    print(f"[cluster] models ({len(model_names)}): " + ", ".join(f"{i}={m}" for i, m in enumerate(model_names)))
    print(f"[cluster] size per dataset (after pad/truncate): {args.num_per_dataset}")

    dataloader = DataLoader(ConcatDataset(router_datasets), batch_size=args.batch_size)
    all_hidden = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(args.device) for k, v in batch[0].items()}
            _, hidden_states = router_model(**inputs)
            all_hidden.append(hidden_states)
    hidden = torch.cat(all_hidden).cpu().numpy()
    print(f"[cluster] embeddings: {hidden.shape}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[cluster] t-SNE ({'openTSNE' if _HAS_OPEN_TSNE else 'sklearn'}, seed={args.seed}) ...")
    if _HAS_OPEN_TSNE:
        low_dim = TSNE(n_components=2, n_jobs=args.tsne_jobs, random_state=args.seed).fit(hidden)
    else:
        low_dim = TSNE(n_components=2, random_state=args.seed).fit_transform(hidden)
    low_dim = np.asarray(low_dim)

    print(f"[cluster] k-means (n_clusters={args.n_clusters}) ...")
    kmeans = KMeans(n_clusters=args.n_clusters, max_iter=1000, random_state=args.seed, n_init=10)
    labels = kmeans.fit(low_dim).labels_.tolist()

    os.makedirs(args.out_dir, exist_ok=True)
    n = args.num_per_dataset
    print(f"  {'dataset':<24}{'written':>9}  cluster ids")
    for i, path in enumerate(dataset_paths):
        # labels start at i * n (padded)
        cluster_ids = labels[i * n:(i + 1) * n]
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        out = []
        for j, sample in enumerate(samples):
            if j >= n:
                break
            sample["cluster_id"] = cluster_ids[j]
            out.append(sample)

        dst = os.path.join(args.out_dir, os.path.basename(path))
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        used = sorted({s["cluster_id"] for s in out})
        print(f"  {os.path.basename(path):<24}{len(out):>9}  {used}")

    print(f"[cluster] done in {time.time() - t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
