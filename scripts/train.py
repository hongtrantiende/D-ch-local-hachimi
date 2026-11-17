#!/usr/bin/env python3
"""Train model dich Trung->Viet kien truc Marian 57M (giong MoxhiMT-60).

Hai che do:
  finetune — load model san co (mac dinh DanVP/MoxhiMT-60, Apache-2.0)
             roi train tiep voi du lieu cua ban (per-series hoac corpus chung).
  scratch  — train tu dau voi config trong configs/ (can corpus lon, GPU nhieu ngay).

Du lieu: JSONL moi dong {"zh": "...", "vi": "..."}.
Tao du lieu bang scripts/prepare_corpus.py hoac scripts/make_series_data.py.
"""

import argparse
import json
import os
import random

from datasets import Dataset
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, MarianConfig,
                          MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments)


def load_jsonl(path):
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            zh, vi = (row.get("zh") or "").strip(), (row.get("vi") or "").strip()
            if zh and vi:
                pairs.append({"zh": zh, "vi": vi})
    return pairs


def build_dataset(pairs, tokenizer, max_len):
    def tok(batch):
        model_in = tokenizer(batch["zh"], truncation=True, max_length=max_len)
        labels = tokenizer(text_target=batch["vi"], truncation=True, max_length=max_len)
        model_in["labels"] = labels["input_ids"]
        return model_in

    ds = Dataset.from_list(pairs)
    return ds.map(tok, batched=True, remove_columns=["zh", "vi"],
                  desc="tokenize")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("finetune", "scratch"), default="finetune")
    ap.add_argument("--init", default="DanVP/MoxhiMT-60",
                    help="model goc de finetune (che do finetune)")
    ap.add_argument("--config", default="configs/marian-zhvi-57m.json",
                    help="config kien truc khi train tu dau")
    ap.add_argument("--tokenizer", default="",
                    help="thu muc tokenizer; de rong = lay tu --init")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="out/model")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None,
                    help="mac dinh: 2e-5 cho finetune, 3e-4 cho scratch")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--max-train", type=int, default=0,
                    help="gioi han so cap cau de test nhanh (0 = khong gioi han)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tokenizer_src = args.tokenizer or args.init
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    train_pairs = load_jsonl(args.train)
    if args.max_train:
        random.seed(args.seed)
        random.shuffle(train_pairs)
        train_pairs = train_pairs[: args.max_train]
    val_pairs = load_jsonl(args.val) if os.path.exists(args.val) else []
    print(f"du lieu: {len(train_pairs)} cap train, {len(val_pairs)} cap val")

    train_ds = build_dataset(train_pairs, tokenizer, args.max_len)
    val_ds = build_dataset(val_pairs, tokenizer, args.max_len) if val_pairs else None

    if args.mode == "finetune":
        model = MarianMTModel.from_pretrained(args.init)
    else:
        with open(args.config, encoding="utf-8") as fh:
            cfg_dict = json.load(fh)
        cfg = MarianConfig(**{k: v for k, v in cfg_dict.items() if not k.startswith("_")})
        model = MarianMTModel(cfg)

    lr = args.lr if args.lr is not None else (2e-5 if args.mode == "finetune" else 3e-4)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="linear",
        fp16=True,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=bool(val_ds),
        generation_max_length=args.max_len,
        report_to=[],
        seed=args.seed,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"da luu model tai {args.out}")


if __name__ == "__main__":
    main()
