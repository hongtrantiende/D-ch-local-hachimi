# Dịch local — train model dịch Trung→Việt chạy trên điện thoại

Project train model dịch **Trung → Việt** kiến trúc **Marian 57M** (cùng kiến
trúc với MoxhiMT-60 / trang moxhi.vietphrase.app) — nhỏ đủ để chạy **ngay trên
máy điện thoại** (ONNX Runtime Mobile) và trên **box InstaBox** (CTranslate2).

Model gốc để fine-tune: [`DanVP/MoxhiMT-60`](https://huggingface.co/DanVP/MoxhiMT-60)
(Apache-2.0, được fine-tune cho võng văn/tiên hiệp).

## Quy trình tổng

```
Bộ truyện (zh/vi lệch dòng) ──make_series_data.py──> train/val.jsonl
                                                        │
              Colab/Kaggle GPU free                     ▼
        train.py --mode finetune ──> out/model ──evaluate.py──> BLEU/chrF
                                                        │
                            ┌───────────────────────────┴────────────┐
                    export_ct2.py                             export_onnx.py
                            ▼                                        ▼
                  out/model-ct2-int8                        out/model-onnx-int8
                  (deploy lên box)                          (đóng gói vào app Android)
```

## Vì sao chọn Marian 57M?

Xem `docs/chon-model.md` — tóm tắt: đã được chứng minh chạy realtime trên CPU
điện thoại (Mặc Hi chạy trong trình duyệt là bằng chứng sống), INT8 chỉ
~35-60 MB, fine-tune từng bộ truyện rẻ, và một model dùng được cho cả box lẫn
máy. NLLB-600M/M2M100 thì quá nặng cho máy phổ thông.

## Train từng bộ một — có khác không?

**Có.** Fine-tune riêng cho từng bộ truyện sẽ cho bản dịch đồng nhất: tên nhân
vật, môn phái, công pháp, xưng hô... giữ nguyên một kiểu suốt bộ. Quy trình
khuyên dùng:

1. **Model gốc chung** = MoxhiMT-60 (hoặc model chung của bạn train trên
   OpenSubtitles, xem `scripts/prepare_corpus.py`).
2. Mỗi bộ truyện tạo thư mục `data/series/<ten-bo>/` với 2 file câu lệch dòng:
   - `<ten-bo>.zh.txt` — văn bản Trung, mỗi dòng một câu
   - `<ten-bo>.vi.txt` — bản dịch Việt tương ứng
3. `python scripts/make_series_data.py` → `train.jsonl` + `val.jsonl` cho từng bộ
4. Fine-tune từ model gốc cho từng bộ (vài phút đến vài giờ GPU free):
   ```bash
   python scripts/train.py --mode finetune \
     --train data/series/ten-bo/train.jsonl \
     --val data/series/ten-bo/val.jsonl \
     --out out/ten-bo --epochs 2
   ```
5. Đánh giá, xuất bản INT8, deploy.

Đừng train từ đầu cho từng bộ — tốn GPU gấp trăm lần và dễ quên hết kiến thức
chung. Train từ đầu (`--mode scratch`) chỉ làm khi muốn model độc lập hoàn
toàn, dùng `scripts/prepare_corpus.py` để lấy vài trăm nghìn cặp câu
OpenSubtitles zh-vi trước.

## Train theo thể loại — 1 model cho mỗi thể loại

Cách làm khuyên dùng: **một model riêng cho một thể loại** (tiên hiệp, đô thị,
linh dị...). Mỗi thể loại có vốn dữ liệu đủ lớn để train ngon, và bản dịch giữ
đúng chất từng thể loại.

**Quy tắc vàng: tất cả model thể loại đều fine-tune từ CÙNG model gốc
MoxhiMT-60.** Đừng bao giờ train thể loại B từ model của thể loại A — kiến
thức sẽ bị trôi dần, càng train nhiều model càng lệch.

```bash
# 1) dữ liệu: data/series/<the-loai>/<the-loai>.zh.txt + .vi.txt (câu lệch dòng)
python scripts/make_series_data.py

# 2) train (luôn từ model gốc, không phải từ model thể loại khác)
python scripts/train.py --mode finetune --init DanVP/MoxhiMT-60 \
  --train data/series/<the-loai>/train.jsonl \
  --val data/series/<the-loai>/val.jsonl \
  --out out/<the-loai> --epochs 2

# 3) đánh giá rồi xuất bản INT8 cho box
python scripts/evaluate.py --model out/<the-loai> --val data/series/<the-loai>/val.jsonl
python scripts/export_ct2.py --model out/<the-loai> --out out/<the-loai>-ct2-int8
```

Deploy lên box: copy `out/<the-loai>-ct2-int8/` vào
`/data/moxhi/genre/<the-loai>/ct2-int8/` trên box. Service tự nhận model mới —
không cần restart, lần gọi đầu tự load.

Gọi API chọn model thể loại:

```json
{"text": "...", "source_lang": "auto", "target_lang": "vi", "model": "tuan-hiep"}
```

Không có trường `"model"` (hoặc `model` rỗng) = dùng model gốc MoxhiMT-60.
`GET /` trên service trả về danh sách thể loại đang có.

## Chạy bằng Colab (không cần máy mạnh)

Mở `colab/train_colab.ipynb` trong Google Colab, chọn GPU (T4 free), làm theo
từng ô. Kết quả `out/model-ct2-int8` + `out/model-onnx` nén lại thành zip.

## Cấu trúc repo

```
configs/          kiến trúc model (57M + bản tiny để smoke-test không cần GPU)
scripts/          prepare_corpus / make_series_data / train / evaluate /
                  train_tokenizer / export_ct2 / export_onnx
colab/            notebook train 1-click
data/series/      chỗ bỏ dữ liệu từng bộ truyện
docs/             chon-model.md, chay-tren-android.md
```

## Deploy lên box (InstaBox)

Sau khi có `out/model-ct2-int8`:

```bash
scp -r out/model-ct2-int8 user@<box>:/data/moxhi/ct2-moi
# trên box:
sudo systemctl stop moxhi
sudo mv /data/moxhi/ct2-int8 /data/moxhi/ct2-cu
sudo mv /data/moxhi/ct2-moi /data/moxhi/ct2-int8
# copy tokenizer của model mới vào /data/moxhi/tokenizer/
sudo systemctl start moxhi
```

Dịch vụ `moxhi` trên box hiện chạy cùng kiến trúc này — thay model là xong,
URL API không đổi.

## Chạy trên điện thoại

Xem `docs/chay-tren-android.md` — ONNX Runtime Mobile, INT8 ~35-60 MB,
chạy CPU không cần GPU, dịch theo đoạn ≤ 300 ký tự để khỏi treo app.

## Ghi chú pháp lý

Model gốc MoxhiMT-60 của DanVP là Apache-2.0 — khi phân phối lại weights
(bundle vào app) cần kèm `LICENSE` và nội dung `NOTICE`.
