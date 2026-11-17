# Chọn model nào để chạy trên điện thoại?

Mục tiêu: model dịch **Trung → Việt** nhỏ đủ để chạy **trên máy điện thoại**
(không cần server), đủ tốt cho văn phong võng văn.

## Bảng so sánh

| Model | Tham số | INT8 trên máy | RAM cần | Tốc độ trên máy | Chất lượng |
|---|---|---|---|---|---|
| **Marian 57M** (kiểu MoxhiMT-60) | ~57M | **~35-60 MB** | ~150 MB | nhanh, realtime | tốt cho võng văn (Mặc Hi là bằng chứng) |
| Marian 6+6 d512 chuẩn | ~74M | ~50-70 MB | ~200 MB | nhanh | tương đương, kiến trúc phổ biến hơn |
| T5-small | ~60M | ~50 MB | ~200 MB | chậm hơn Marian ~2x | cần dữ liệu nhiều hơn để ngang Marian |
| M2M100-418M | 418M | ~230 MB | ~700 MB | chậm | đa ngôn ngữ, lãng phí cho 1 cặp ngôn ngữ |
| NLLB-200-600M | 600M | ~300 MB | ~1 GB | rất chậm, máy nóng | tốt nhất đa ngôn ngữ nhưng quá nặng cho máy phổ thông |

## Chọn: Marian 57M (8 tầng encoder + 2 tầng decoder, d_model 576)

Lý do:

1. **Đã được chứng minh thực tế**: trang moxhi.vietphrase.app chạy đúng kiến trúc này
   ngay trong trình duyệt điện thoại bằng WebGPU — tức CPU điện thoại chạy nổi.
2. **Cả repo này lẫn box đều dùng chung được**: cùng xuất bản CTranslate2 INT8
   cho box và ONNX INT8 cho Android, không phải giữ hai model khác nhau.
3. **Fine-tune từng bộ truyện rẻ**: 57M tham số, vài nghìn cặp câu là thấy khác.
4. License model gốc DanVP/MoxhiMT-60 là **Apache-2.0** — được phép fine-tune
   và phân phối lại (ghi NOTICE theo yêu cầu của tác giả).

## Hai mức train

- **Fine-tune từng bộ** (khuyên dùng): train tiếp từ model gốc hoặc từ model
  chung của bạn, mỗi bộ vài giờ GPU free. Model của từng bộ sẽ gọi tên nhân vật,
  thuật ngữ đồng nhất.
- **Train từ đầu**: chỉ khi muốn độc lập hoàn toàn, cần hàng triệu cặp câu
  (OPUS/OpenSubtitles zh-vi ...) và GPU chạy nhiều ngày.

## Quyết định (đã đo thật trên box)

Đã benchmark `MoxhiMT-60` vs `MoxhiMT-30` (cùng tác giả, cùng tokenizer family)
trên box 4 vCPU với 5 mẫu võng văn (2026-11-17):

| | 60M (ct2-int8) | 30M (ct2-int8_float32) |
|---|---|---|
| Chất lượng 5 mẫu | ngang nhau | ngang nhau |
| Tốc độ câu ngắn | ~0.35s | ~0.25s |
| Tốc độ câu dài | ~0.65s | ~0.39s |
| Kích thước model.bin | 58 MB | 38 MB |

Chọn **MoxhiMT-60 làm model chuẩn để train** vì:

1. Fine-tune từng bộ cần "dung lượng" — 57M tham số tiếp thu thuật ngữ của
   nhiều bộ truyện mà không làm hỏng kiến thức chung; 30M dễ bão hòa hơn.
2. Đã deploy sẵn trên box, tokenizer đã cài, đã tối ưu beam.
3. Điện thoại vẫn chạy tốt 60M (trang Mặc Hi chạy được trong trình duyệt).

MoxhiMT-30 đã tải về box (`/data/moxhi/ct2-30` + `/data/moxhi/tokenizer-30`)
để làm **biến thể nhanh tùy chọn** — nếu sau này cần ưu tiên tốc độ/điện thoại
yếu, chỉ cần đổi đường dẫn model trong service, không phải đổi pipeline.

