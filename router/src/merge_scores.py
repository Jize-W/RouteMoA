"""Usage:
    python src/merge_scores.py --weight 0.5 \
        --int_dir data/int_score/train \
        --reward_dir data/reward_score/train \
        --out_dir data/merged/train_lambda_0.5
"""

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


def question_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text.strip().lower()


def blend(a: float, b: float, w: float) -> float:
    return max(0.0, min(1.0, a * w + b * (1.0 - w)))


def load_json_list(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array at top level")
    return data


def index_by_question(items: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for rec in items:
        q = rec.get("question")
        if q is None:
            continue
        out.setdefault(question_key(q), rec)
    return out


def list_json(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".json")


def merge_pair(path_a: Path, path_b: Path, w: float) -> Tuple[List[Dict], int, int]:
    items_a = load_json_list(path_a)
    items_b = load_json_list(path_b)
    idx_b = index_by_question(items_b)

    merged: List[Dict] = []
    for rec_a in items_a:
        key = question_key(rec_a["question"])
        rec_b = idx_b.get(key)
        if rec_b is None:
            continue

        scores_a = rec_a.get("scores", {})
        scores_b = rec_b.get("scores", {})
        scores = {}
        for model, score_a in scores_a.items():
            if model not in scores_b:
                continue
            scores[model] = blend(float(score_a), float(scores_b[model]), w)
        if not scores:
            continue
        merged.append({"question": rec_a["question"], "scores": scores})

    return merged, len(items_a), len(merged)


def reward_filename(name: str) -> List[str]:
    # "-train.json" vs "_train.json"
    p = Path(name)
    cands = [name]
    if "-" in p.stem:
        ds, _, split = p.stem.rpartition("-")
        cands.append(f"{ds}_{split}{p.suffix}")
    return cands


def merge_dirs(dir_a: Path, dir_b: Path, out_dir: Path, w: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for path_a in list_json(dir_a):
        path_b = None
        for cand in reward_filename(path_a.name):
            if (dir_b / cand).is_file():
                path_b = dir_b / cand
                break
        if path_b is None:
            print(f"  [skip] no reward file for {path_a.name}")
            continue
        pairs.append((path_a, path_b))

    if not pairs:
        raise SystemExit(f"No pairable files between {dir_a} and {dir_b}")

    total_in = total_out = 0
    print(f"  {'dataset':<24}{'records':>9}{'matched':>9}{'kept':>8}")
    for path_a, path_b in pairs:
        merged, n_a, n_out = merge_pair(path_a, path_b, w)
        total_in += n_a
        total_out += n_out
        with (out_dir / path_a.name).open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        print(f"  {path_a.name:<24}{n_a:>9}{n_out:>9}{n_out / max(n_a, 1):>8.1%}")

    print(f"  {'TOTAL':<24}{total_in:>9}{total_out:>9}{total_out / max(total_in, 1):>8.1%}")
    print(f"  -> {out_dir}  (lambda = {w:g})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="merge int_score and reward_score into lambda-blended labels")
    p.add_argument("--int_dir", default="data/int_score/train", help="directory of 0/1 correctness json files")
    p.add_argument("--reward_dir", default="data/reward_score/train", help="directory of reward-score json files")
    p.add_argument("--out_dir", default="data/merged/train_lambda_0.5", help="output directory")
    p.add_argument("--weight", "-w", type=float, default=0.5,
                   help="lambda in Eq.(4): weight of the 0/1 correctness term (1-w goes to the reward term)")
    args = p.parse_args(argv)

    if not 0.0 <= args.weight <= 1.0:
        p.error("--weight must be between 0 and 1")

    print(f"[merge_scores] lambda={args.weight:g}")
    print(f"  int_score    : {args.int_dir}")
    print(f"  reward_score : {args.reward_dir}")
    merge_dirs(Path(args.int_dir), Path(args.reward_dir), Path(args.out_dir), args.weight)


if __name__ == "__main__":
    main(sys.argv[1:])
