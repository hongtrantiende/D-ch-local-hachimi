#!/usr/bin/env python3
"""Tu dong dich tap chuong Trung bang endpoint dang DeepLX de tao bo du lieu
song ngu cho fine-tune the loai (domain self-adaptation).

  python3 self_translate.py \
    --chapters-dir /data/novels/tien-hiep/zh \
    --endpoint http://localhost:1199/translate \
    --out-dir /data/series-data/tien-hiep \
    --translated-dir /data/novels/tien-hiep/vi \
    --workers 3 --max-chars 350 --val-ratio 0.02

Luu y: day la self-training (ban dich do chinh model sinh ra) — model hoc van
phong + dong nhat thuat ngu the loai, KHONG co tri thuc moi. Co ban dich that
thi train lai bang du lieu do.
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def translate(endpoint, text, retries=3):
    body = json.dumps({"text": text, "source_lang": "auto",
                       "target_lang": "vi"}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(endpoint, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                doc = json.loads(r.read())
            return doc.get("data") or ""
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  ! loi sau {retries} lan: {exc}")
                return ""
            time.sleep(2 * (attempt + 1))


def chunk_text(text, max_chars):
    """Gom cac doan lien ke thanh chunk <= max_chars, giu xuong dong."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur and cur_len + len(p) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters-dir", required=True)
    ap.add_argument("--endpoint", default="http://localhost:1199/translate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--translated-dir", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=350)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.chapters_dir)
                   if f.endswith(".txt") and not f.startswith("0000"))
    files.sort(key=lambda f: int(re.match(r"(\d+)", f).group(1)) if re.match(r"\d", f) else 0)
    print(f"{len(files)} chuong")

    jobs = []  # (file, title, chunk_idx, chunk_text)
    for fname in files:
        raw = open(os.path.join(args.chapters_dir, fname),
                   encoding="utf-8").read().replace("\r", "")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if not lines:
            continue
        title, body = lines[0], "\n".join(lines[1:])
        jobs.append((fname, title, 0, title))
        for i, chunk in enumerate(chunk_text(body, args.max_chars)):
            if len(chunk) < 5:
                continue
            jobs.append((fname, title, i + 1, chunk))
    print(f"{len(jobs)} phan can dich")

    results = {}  # (fname, idx) -> vi
    done = [0]

    def work(job):
        fname, title, idx, zh = job
        vi = translate(args.endpoint, zh)
        results[(fname, idx)] = vi
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  {done[0]}/{len(jobs)} phan")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    dt = time.time() - t0
    print(f"dich xong {len(results)} phan trong {dt/60:.1f} phut")

    # ghep chuong dich va tap hop cap cau
    os.makedirs(args.translated_dir or "/tmp/x", exist_ok=True) if args.translated_dir else None
    chapters = {}
    for fname, title, idx, zh in jobs:
        chapters.setdefault(fname, {})[idx] = (zh, results.get((fname, idx), ""))
    pairs = []
    for fname in sorted(chapters):
        if args.translated_dir:
            vi_lines = []
            for idx in sorted(chapters[fname]):
                zh, vi = chapters[fname][idx]
                if vi:
                    vi_lines.append(vi)
            if vi_lines:
                with open(os.path.join(args.translated_dir, fname), "w",
                          encoding="utf-8") as fh:
                    fh.write("\n\n".join(vi_lines) + "\n")
        for idx in sorted(chapters[fname]):
            zh, vi = chapters[fname][idx]
            if vi and len(vi) >= 3:
                pairs.append({"zh": zh, "vi": vi})
    print(f"{len(pairs)} cap cau")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train.jsonl", pairs[n_val:]), ("val.jsonl", pairs[:n_val])):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"xong: {len(pairs) - n_val} train, {n_val} val -> {args.out_dir}/")


if __name__ == "__main__":
    main()
