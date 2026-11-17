#!/usr/bin/env python3
"""Tai corpus song ngu Trung-Viet cong khai (OpenSubtitles qua HuggingFace),
lam sach va tao data/train.jsonl + data/val.jsonl.

Chay can internet, gan GPU khong bat buoc (chi can tai du lieu).
  python scripts/prepare_corpus.py --n 400000
  python scripts/prepare_corpus.py --subset vi-zh_cn --n 400000
"""

import argparse
import json
import random


def clean(zh, vi):
    zh, vi = zh.strip(), vi.strip()
    if not zh or not vi:
        return None
    if len(zh) < 2 or len(vi) < 2:
        return None
    if len(zh) > 200 or len(vi) > 300:
        return None
    # ti le do dai qua lech: cap cau lech dang
    if not (0.4 <= len(vi) / max(len(zh), 1) <= 4.0):
        return None
    for bad in ("<i>", "</i>", "♪", "－"):
        if bad in zh or bad in vi:
            return None
    return {"zh": zh, "vi": vi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="vi-zh_cn",
                    help="config cua Helsinki-NLP/open_subtitles (vi-zh_cn, vi-zh_Hans...)")
    ap.add_argument("--n", type=int, default=400000, help="so cap cau lay ra")
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"tai {args.subset} (streaming)...")
    ds = load_dataset("Helsinki-NLP/open_subtitles", args.subset,
                      streaming=True, split="train")
    seen, pairs = set(), []
    for ex in ds:
        tr = ex.get("translation") or {}
        zh = tr.get("zh_cn") or tr.get("zh") or tr.get("zh_Hans") or ""
        vi = tr.get("vi") or ""
        row = clean(zh, vi)
        if not row or row["zh"] in seen:
            continue
        seen.add(row["zh"])
        pairs.append(row)
        if len(pairs) >= args.n * 2:  # chieu hoi de con loc ty le
            break
        if len(pairs) % 20000 == 0 and len(pairs):
            print(f"  {len(pairs)} cap")
    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[: args.n]
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val, train = pairs[:n_val], pairs[n_val:]
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(f"{args.out_dir}/{name}", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"xong: {len(train)} train, {len(val)} val -> {args.out_dir}/")


if __name__ == "__main__":
    main()
