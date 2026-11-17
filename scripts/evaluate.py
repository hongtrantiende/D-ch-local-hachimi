#!/usr/bin/env python3
"""Do BLEU/chrF cua model moi so voi model goc tren tap val.

Chay tot nhat tren GPU. Vi du:
  python scripts/evaluate.py --model out/model --base DanVP/MoxhiMT-60 \
      --val data/val.jsonl --n 300
"""

import argparse
import json

import sacrebleu
import torch
from transformers import AutoTokenizer, MarianMTModel


def load_pairs(path, n):
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pairs.append((row["zh"], row["vi"]))
            if len(pairs) >= n:
                break
    return pairs


def translate(model, tokenizer, texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=320).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=320, num_beams=4)
        out += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return out


def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return bleu.score, chrf.score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model moi (thu muc out/)")
    ap.add_argument("--base", default="DanVP/MoxhiMT-60")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip-base", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs(args.val, args.n)
    srcs = [p[0] for p in pairs]
    refs = [p[1] for p in pairs]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    hyps = translate(model, tokenizer, srcs)
    bleu, chrf = score(hyps, refs)
    print(f"model moi  : BLEU {bleu:.2f} | chrF {chrf:.2f}")

    if not args.skip_base and args.base != args.model:
        tok_b = AutoTokenizer.from_pretrained(args.base)
        model_b = MarianMTModel.from_pretrained(args.base).eval()
        if torch.cuda.is_available():
            model_b = model_b.cuda()
        hyps_b = translate(model_b, tok_b, srcs)
        bleu_b, chrf_b = score(hyps_b, refs)
        print(f"model goc  : BLEU {bleu_b:.2f} | chrF {chrf_b:.2f}")


if __name__ == "__main__":
    main()
