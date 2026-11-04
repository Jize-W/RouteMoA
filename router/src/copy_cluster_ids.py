"""Usage:
    python src/copy_cluster_ids.py \
        --cluster_dir data/clustered/train_lambda_0.5 \
        --merged_dir  data/merged/train_lambda_0.3 \
        --out_dir     data/clustered/train_lambda_0.3
"""

import argparse
import json
import os
import unicodedata
from pathlib import Path


def question_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split()).strip().lower()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="propagate cluster_id to another lambda split")
    p.add_argument("--cluster_dir", required=True, help="clustered split that already has cluster_id")
    p.add_argument("--merged_dir", required=True, help="merged split to annotate (question + scores)")
    p.add_argument("--out_dir", required=True, help="output directory")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src_dir, dst_dir, out_dir = Path(args.cluster_dir), Path(args.merged_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {'dataset':<24}{'written':>9}  matched")
    total = 0
    for src in sorted(src_dir.glob("*.json")):
        dst = dst_dir / src.name
        if not dst.is_file():
            print(f"  [skip] {src.name}: not found in {dst_dir}")
            continue

        cluster_of = {question_key(r["question"]): r["cluster_id"] for r in json.load(src.open(encoding="utf-8"))}
        records = json.load(dst.open(encoding="utf-8"))

        out, missing = [], 0
        for rec in records:
            cid = cluster_of.get(question_key(rec["question"]))
            if cid is None:
                missing += 1
                continue
            rec["cluster_id"] = cid
            out.append(rec)

        with (out_dir / src.name).open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        total += len(out)
        if missing:
            print(f"  {src.name:<24}{len(out):>9}  {len(out)}/{len(records)}  (missing {missing}, the part where merged has more than clustered)")
        else:
            print(f"  {src.name:<24}{len(out):>9}  all")

    if total == 0:
        raise SystemExit("nothing written -- check --cluster_dir / --merged_dir")
    print(f"  {'TOTAL':<24}{total:>9}")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
