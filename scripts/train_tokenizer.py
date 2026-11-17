#!/usr/bin/env python3
"""Train tokenizer SentencePiece-BPE joint ZH+VI (chi dung khi train tu dau).

Fine-tune tu MoxhiMT-60 KHONG can buoc nay (dung lai tokenizer goc).
  python scripts/train_tokenizer.py --data data/train.jsonl --out tokenizer-zhvi
"""

import argparse
import json

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--vocab", type=int, default=24000)
    ap.add_argument("--out", default="tokenizer-zhvi")
    args = ap.parse_args()

    tmp = args.out + "_corpus.txt"
    with open(tmp, "w", encoding="utf-8") as fh:
        with open(args.data, encoding="utf-8") as src:
            for line in src:
                row = json.loads(line)
                fh.write(row["zh"] + "\n" + row["vi"] + "\n")

    spm.SentencePieceTrainer.train(
        input=tmp,
        model_prefix=args.out + "/spm",
        vocab_size=args.vocab,
        model_type="bpe",
        character_coverage=0.99995,
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
    print(f"tokenizer da luu tai {args.out}/ (spm.model)")


if __name__ == "__main__":
    main()
