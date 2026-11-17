#!/usr/bin/env python3
"""Xuat model da train sang CTranslate2 INT8 — de chay tren box InstaBox.

  python scripts/export_ct2.py --model out/model --out out/model-ct2-int8

Sau do deploy len box (thay thu muc /data/moxhi/ct2-int8 va restart service):
  scp -r out/model-ct2-int8 user@box:/data/moxhi/
  # tren box: doi ten, restart systemctl moxhi
Luu y: dich vu moxhi tren box doc tokenizer tu thu muc /data/moxhi/tokenizer,
nen cung copy cac file tokenizer (vocab.json, spm) vao model moi.
"""

import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/model")
    ap.add_argument("--out", default="out/model-ct2-int8")
    args = ap.parse_args()

    from ctranslate2.converters import TransformersConverter

    converter = TransformersConverter(args.model, load_as_float16=True)
    converter.convert(args.out, quantization="int8")

    # copy tokenizer can thiet cho dich vu tren box
    for name in ("source.spm", "target.spm", "vocab.json",
                 "tokenizer_config.json", "special_tokens_map.json",
                 "shared_vocabulary.json"):
        src = os.path.join(args.model, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
    print(f"xong: {args.out}")


if __name__ == "__main__":
    main()
