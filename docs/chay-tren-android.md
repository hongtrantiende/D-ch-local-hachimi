# Chạy model trên Android

Model xuất ONNX INT8 từ `scripts/export_onnx.py` chạy được trên máy bằng
**ONNX Runtime Mobile** (thư viện ~10 MB, CPU, không cần GPU/NPU).

## Kích thước dự kiến

| Bản | Kích thước | Ghi chú |
|---|---|---|
| fp32 ONNX | ~230 MB | chỉ dùng khi debug |
| **INT8 ONNX** | **~35-60 MB** | bản đóng gói vào app |
| CTranslate2 INT8 | ~58 MB | bản cho box (server), không dùng trên máy |

## Tích hợp vào app (Novela MD3)

1. Thêm dependency `onnxruntime-android` (com.microsoft.onnxruntime).
2. Đóng gói file `encoder.onnx` + `decoder.onnx` + tokenizer vào `assets/`
   (hoặc tải lần đầu về `filesDir/` để khỏi phình APK).
3. Luồng dịch: tokenizer (SentencePiece qua JNI hoặc bản Java) → encoder →
   decoder từng bước (beam 1 đủ nhanh trên máy, beam 4 nếu máy khỏe).
4. Nên dịch **từng đoạn ngắn** (≤ 300 ký tự) và hiển thị dần, tránh treo UI:
   chạy trong coroutine `Dispatchers.Default`.

## Ghi chú

- Tokenizer SentencePiece-BPE 24k dùng chung cho cả hai phía (joint).
- Nếu máy yếu: dùng bản INT8 + beam 1 + max_length 256.
- Bản MoxhiMT-30 (30M, nhỏ hơn nữa) là phương án dự phòng nếu 60M vẫn chậm
  trên máy cấu hình thấp — cùng pipeline xuất.
