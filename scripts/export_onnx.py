#!/usr/bin/env python3
"""Xuat model sang ONNX (+ quantize INT8) de chay tren dien thoai Android
qua ONNX Runtime Mobile.

  python scripts/export_onnx.py --model out/model --out out/model-onnx

Ket qua: out/model-onnx/ chua encoder.onnx, decoder.onnx (INT8) + tokenizer.
Kich thuoc du kien: ~35-60 MB — chay du tren CPU dien thoai.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/model")
    ap.add_argument("--out", default="out/model-onnx")
    args = ap.parse_args()

    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    model = ORTModelForSeq2SeqLM.from_pretrained(args.model, export=True)
    model.save_pretrained(args.out)
    model.tokenizer.save_pretrained(args.out)

    # quantize dong INT8 — kich thuoc nho hon ~4 lan, nhanh tren CPU ARM
    quantizer = ORTQuantizer.from_pretrained(args.out)
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=args.out + "-int8",
                       quantization_config=qconfig)
    print(f"xong: {args.out} (fp32) va {args.out}-int8 (quantized)")


if __name__ == "__main__":
    main()
