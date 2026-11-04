"""Usage:
    python src/reward_score.py --input responses.jsonl \
        --out data/reward_score/train/math_train.json \
        --judger internlm/internlm2-1_8b-reward
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="score responses with a reward model")
    p.add_argument("--input", required=True, help="input jsonl, one record per line")
    p.add_argument("--out", required=True, help="output json (list of {question, scores})")
    p.add_argument("--judger", default="internlm/internlm2-1_8b-reward",
                   help="reward model: HF repo id, or a local directory")
    p.add_argument("--models", nargs="+", default=None,
                   help="if set, only score these model names (default: every key found)")
    p.add_argument("--limit", type=int, default=0, help="only score the first N records (0 = all)")
    p.add_argument("--offset", type=int, default=0, help="skip the first N records")
    p.add_argument("--max_len", type=int, default=0,
                   help="skip records whose reward input exceeds this many tokens (0 = no limit)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                   help="bfloat16 matches the shipped scores")
    p.add_argument("--with_responses", action="store_true",
                   help="also store the scored response next to each score")
    p.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                   help="rename an output key, repeatable: --rename OLD=NEW --rename OLD2=NEW2")
    p.add_argument("--question_field", default="question",
                   help="field used as the output 'question' key (must match int_score); "
                        "falls back to 'prompt' when absent")
    p.add_argument("--compare", default=None,
                   help="an existing reward_score json, compared record by record after scoring")
    p.add_argument("--flush_every", type=int, default=50,
                   help="write the partial output every N records (protects long runs)")
    return p.parse_args(argv)


def extract(prompt_obj):
    if isinstance(prompt_obj.get("responses"), dict):
        return {str(k): str(v) for k, v in prompt_obj["responses"].items()}
    models = prompt_obj.get("models")
    preds = prompt_obj.get("origin_prediction")
    if isinstance(models, dict) and isinstance(preds, dict):
        # keys differ, so pair by position
        names = [str(v) for v in models.values()]
        values = list(preds.values())
        return {n: str(r) for n, r in zip(names, values) if r is not None}
    if isinstance(models, list) and isinstance(preds, list):
        return {str(n): str(r) for n, r in zip(models, preds)}
    raise ValueError(
        "cannot find responses: expected either a 'responses' dict, or "
        "'models' + 'origin_prediction' (dict or list)"
    )


def load_records(path, offset, limit, question_field="question"):
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("prompt", obj.get("question"))
            if text is None:
                raise ValueError(f"{path}: record {i} has neither 'prompt' nor 'question'")
            question = obj.get(question_field) or text
            records.append({"index": i, "question": question, "text": text,
                            "responses": extract(obj)})
    return records


def token_len(tokenizer, prompt, response):
    chat = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


def dump(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    os.replace(tmp, path)


def compare(rows, archive_path, models):
    with open(archive_path, encoding="utf-8") as f:
        arch = json.load(f)

    def q(rec):
        v = rec.get("question", rec.get("prompt"))
        return v.strip() if isinstance(v, str) else None

    def scores_of(rec):
        if isinstance(rec.get("scores"), dict):
            return rec["scores"]
        return {k: v["score"] for k, v in rec.items()
                if isinstance(v, dict) and "score" in v}

    by_index = sum(1 for r in rows
                   if r["index"] < len(arch) and q(arch[r["index"]]) == r["question"].strip())
    use_index = by_index >= max(1, len(rows) // 2)
    lookup = {}
    if not use_index:
        for rec in arch:
            lookup.setdefault(q(rec), rec)

    matched = 0
    worst = 0.0
    n_diff = 0
    for row in rows:
        if use_index:
            i = row["index"]
            rec = arch[i] if i < len(arch) else None
            if rec is None or q(rec) != row["question"].strip():
                continue
        else:
            rec = lookup.get(row["question"].strip())
            if rec is None:
                continue
        matched += 1
        ref = scores_of(rec)
        for m in models:
            if m not in row["scores"] or m not in ref:
                continue
            d = abs(row["scores"][m] - ref[m])
            worst = max(worst, d)
            if d > 1e-6:
                n_diff += 1
    print("\n================ compare with archive ================")
    print(f"  alignment         : {'row index' if use_index else 'question key'}")
    print(f"  records matched   : {matched}")
    print(f"  score pairs > 1e-6: {n_diff}")
    print(f"  max |delta|       : {worst:.6e}")
    if matched:
        print("  (small deltas of ~1e-2 are expected: bf16 + different GPU kernels)")


def main(argv=None):
    args = parse_args(argv)
    print(f"[info] torch {torch.__version__} | cuda {torch.cuda.is_available()} "
          f"| HF_HOME={os.environ.get('HF_HOME')} | HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")

    records = load_records(args.input, args.offset, args.limit, args.question_field)
    if not records:
        raise SystemExit(f"[error] no records read from {args.input}")
    found = []
    for r in records:
        for m in r["responses"]:
            if m not in found:
                found.append(m)
    models = args.models or found
    if args.models:
        for r in records:
            r["responses"] = {m: v for m, v in r["responses"].items() if m in models}
    rename = {}
    for item in args.rename:
        old, _, new = item.partition("=")
        if not new:
            raise SystemExit(f"[error] --rename expects OLD=NEW, got {item!r}")
        rename[old] = new
    print(f"[info] {len(records)} records | {len(models)} models | "
          f"{sum(len(r['responses']) for r in records)} scoring calls")
    if rename:
        print(f"[info] renaming {len(rename)} output keys")

    t0 = time.time()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    judger = AutoModel.from_pretrained(
        args.judger, device_map=0, torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.judger, trust_remote_code=True, model_max_length=131072)
    print(f"[info] reward model loaded in {time.time() - t0:.1f}s | dtype={judger.dtype}")

    rows = []
    skipped = 0
    t0 = time.time()
    for n, rec in enumerate(records, 1):
        scores = {}
        details = {}
        for model, response in rec["responses"].items():
            if args.max_len and token_len(tokenizer, rec["text"], response) > args.max_len:
                skipped += 1
                print(f"  [skip] record {rec['index']} / {model}: too long", flush=True)
                continue
            chat = [{"role": "user", "content": rec["text"]},
                    {"role": "assistant", "content": response}]
            with torch.inference_mode():
                score = judger.get_score(tokenizer, chat)
            scores[rename.get(model, model)] = float(score)
            details[rename.get(model, model)] = {"response": response, "score": float(score)}
        if not scores:
            continue
        row = {"index": rec["index"], "question": rec["question"], "scores": scores}
        if args.with_responses:
            row["models"] = details
        rows.append(row)
        if n % 10 == 0 or n == len(records):
            el = time.time() - t0
            print(f"  [{n}/{len(records)}] {el:.1f}s ({el / n * 1000:.0f} ms/record)", flush=True)
        if args.flush_every and n % args.flush_every == 0:
            dump(args.out, rows)

    dump(args.out, rows)
    print(f"[done] {len(rows)} records -> {args.out} | skipped {skipped} | "
          f"{time.time() - t0:.1f}s for {len(rows)} records")
    if args.compare:
        compare(rows, args.compare, [rename.get(m, m) for m in models])


if __name__ == "__main__":
    main()
