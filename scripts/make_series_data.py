#!/usr/bin/env python3
"""Ghep du lieu tung bo truyen thanh cap cau Trung-Viet de fine-tune.

Bo vao data/series/<ten-bo>/ hai file can le theo dong:
  <ten-bo>.zh.txt  van ban Trung, moi dong mot cau
  <ten-bo>.vi.txt  ban dich Viet tuong ung, moi dong mot cau

Xuat ra data/series/<ten-bo>/train.jsonl + val.jsonl (95/5).
Lap lai cho moi bo: moi bo mot model rieng, hoac gom nhieu bo de train chung:
  python scripts/make_series_data.py --merge-all
"""

import argparse
import json
import os
import random


def read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n").strip() for line in fh]


def collect_dir(series_dir):
    zh_files = [f for f in os.listdir(series_dir) if f.endswith(".zh.txt")]
    if not zh_files:
        return []
    zh_path = os.path.join(series_dir, zh_files[0])
    vi_path = zh_path.replace(".zh.txt", ".vi.txt")
    if not os.path.exists(vi_path):
        raise SystemExit(f"thieu file dich: {vi_path}")
    zh_lines, vi_lines = read_lines(zh_path), read_lines(vi_path)
    if len(zh_lines) != len(vi_lines):
        print(f"! {series_dir}: so dong lech ({len(zh_lines)} zh / {len(vi_lines)} vi), "
              f"bo qua cac dong thua")
    pairs, skipped = [], 0
    for zh, vi in zip(zh_lines, vi_lines):
        zh, vi = zh.strip(), vi.strip()
        if not zh or not vi:
            skipped += 1
            continue
        if len(zh) > 400 or len(vi) > 600:
            skipped += 1
            continue
        pairs.append({"zh": zh, "vi": vi})
    if skipped:
        print(f"  bo qua {skipped} dong rong/qua dai")
    return pairs


def write_pairs(pairs, out_dir, val_ratio, seed):
    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 20 else 0
    val, train = pairs[:n_val], pairs[n_val:]
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(train), len(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/series")
    ap.add_argument("--merge-all", action="store_true",
                    help="gom moi bo thanh mot tap train/val chung")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--min-pairs", type=int, default=50,
                    help="so cap toi thieu de bo qua bo qua (bo qua xu ly)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    series = sorted(d for d in os.listdir(args.root)
                    if os.path.isdir(os.path.join(args.root, d))
                    and not d.startswith("."))
    if not series:
        raise SystemExit(f"khong co bo nao trong {args.root}")

    if args.merge_all:
        all_pairs = []
        for name in series:
            all_pairs += collect_dir(os.path.join(args.root, name))
        n_train, n_val = write_pairs(all_pairs, os.path.join(args.root, "all"),
                                     args.val_ratio, args.seed)
        print(f"[all] {n_train} train, {n_val} val")
    else:
        for name in series:
            pairs = collect_dir(os.path.join(args.root, name))
            if len(pairs) < args.min_pairs:
                print(f"[{name}] chi co {len(pairs)} cap — qua it de train, bo qua")
                continue
            n_train, n_val = write_pairs(pairs, os.path.join(args.root, name),
                                         args.val_ratio, args.seed)
            print(f"[{name}] {n_train} train, {n_val} val")


if __name__ == "__main__":
    main()
