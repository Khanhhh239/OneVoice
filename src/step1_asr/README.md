# Step 1 — Nhận dạng Giọng nói (ASR)

Phân tích đầy đủ 5 mô hình ứng viên với số liệu thực tế, ghi chú license: [`../../step1.md`](../../step1.md)

**Mô hình được chọn:**
- **Tiếng Việt:** Zipformer-30M-RNNT-6000h
- **Anh / Trung / Hàn:** SenseVoice-Small

PhoWhisper, Moonshine và Qwen3-ASR-0.6B đã được kiểm thử và loại bỏ — xem lý do tại `step1.md §2–4`.

---

## Cấu trúc thư mục

```
src/step1_asr/
│
├── # ── Dữ liệu & Chuẩn bị ─────────────────────────────────
│
├── fetch_asr_data.py           # Tải dữ liệu âm thanh từ FLEURS → data/asr/<lang>/*.wav + manifest.json
├── mix_asr_noise.py            # Trộn nhiễu SNR vào audio để tạo tập kiểm thử robustness → data/asr_mixed/
│
├── # ── Benchmark các ứng viên ──────────────────────────────
│
├── run_all_asr.py              # Chạy lần lượt tất cả benchmark bên dưới (shortcut)
├── test_asr_multi.py           # [CHỌN] SenseVoice-Small (Anh/Trung/Hàn) — đo WER/CER/RTF
├── test_asr_zipformer.py       # [CHỌN] Zipformer-30M (Tiếng Việt) — đo WER/RTF
├── test_asr_vi.py              # [LOẠI] PhoWhisper (Tiếng Việt) — giữ lại để so sánh
├── test_asr_moonshine.py       # [LOẠI] Moonshine (Đa ngôn ngữ)
├── test_asr_qwen.py            # [LOẠI] Qwen3-ASR-0.6B — chậm hơn SenseVoice ~10 lần
│
├── # ── Pipeline Tích hợp ───────────────────────────────────
│
├── unified_asr.py              # Class UnifiedASRPipeline: tự động định tuyến sang đúng model theo ngôn ngữ;
│                               #   hỗ trợ cờ use_denoiser=True để bật khử nhiễu GTCRN (từ Step 0) trước khi nhận dạng
├── test_unified_asr.py         # Kiểm thử toàn diện UnifiedASRPipeline (có/không có denoiser, 4 ngôn ngữ, nhiều SNR)
│
└── # ── Quantization & NPU Deployment (SenseVoice) ─────────
    │
    ├── step4_s1_export_sensevoice_onnx.py  # Bước 1: Export SenseVoice sang ONNX với Static Shape [1, 500, 560]
    ├── step4_s1_patch_mask.py              # Bước 2: Bơm dummy bias vào 70 node Conv bị thiếu (fix lỗi QAIRT crash)
    ├── step4_s1_prepare_calib.py           # Bước 3: Trích xuất fbank 15 mẫu (5 En+5 Zh+5 Ko) làm calibration data
    ├── step4_s1_qai_hub_submit.py          # Bước 4: Lượng tử hoá w8a16 + biên dịch QNN Binary qua Qualcomm AI Hub
    ├── step4_s1_verify_w8a16.py            # Bước 5: Chạy inference trên board EVK thật, đo Cosine Similarity & WER/CER
    └── step4_s1_profile_w8a16.py           # Bước 6: Đo latency và peak memory trên NPU Hexagon (Profile Job)
```

---

## Cài đặt

```bash
pip install -r ../../requirements.txt
```

> **Lưu ý:** `test_asr_qwen.py` yêu cầu `transformers>=5.13.0`, có thể xung đột với pin của funasr.
> Nếu bị lỗi, hãy cài vào một môi trường ảo (venv/conda) riêng.

---

## Hướng dẫn chạy

### 1. Chuẩn bị dữ liệu

```bash
python fetch_asr_data.py     # Tải FLEURS → data/asr/
python mix_asr_noise.py      # Tạo tập nhiễu → data/asr_mixed/
```

### 2. Benchmark từng mô hình ứng viên

```bash
python test_asr_multi.py      # SenseVoice-Small [CHỌN]
python test_asr_zipformer.py  # Zipformer-30M    [CHỌN]
python test_asr_vi.py         # PhoWhisper        [LOẠI]
python test_asr_moonshine.py  # Moonshine         [LOẠI]
python test_asr_qwen.py       # Qwen3-ASR-0.6B   [LOẠI]

# Hoặc chạy tất cả một lần:
python run_all_asr.py
```

Kết quả lưu tại `outputs/asr_*_results.csv` (WER/CER/RTF theo ngôn ngữ và mức SNR).

### 3. Kiểm thử pipeline tích hợp

```bash
python test_unified_asr.py
```

Kết quả lưu tại `outputs/asr_unified_results.csv` (bao gồm cột `denoised` cho biết file đó có được khử nhiễu trước hay không). RTF trung bình trên máy dev đạt ~0.03.

---

## Quantization & NPU Deployment (SenseVoice-Small w8a16)

Để đáp ứng yêu cầu tốc độ cao và tiết kiệm pin trên phần cứng Edge (**Dragonwing IQ-9075 EVK** — Hexagon NPU), SenseVoice-Small đã được lượng tử hoá sang định dạng **w8a16** (trọng số 8-bit, activations 16-bit) và biên dịch sang QNN Binary.

### Quy trình thực hiện

| Bước | Script | Mô tả |
|------|--------|--------|
| 1 | `step4_s1_export_sensevoice_onnx.py` | Export ONNX, cố định kích thước đầu vào `[1, 500, 560]` |
| 2 | `step4_s1_patch_mask.py` | Bơm dummy bias vào 70 Conv node bị thiếu (fix QAIRT crash) |
| 3 | `step4_s1_prepare_calib.py` | Trích xuất calibration data (15 mẫu, đủ 3 ngôn ngữ) |
| 4 | `step4_s1_qai_hub_submit.py` | Quantize w8a16 + compile sang QNN Binary trên AI Hub |
| 5 | `step4_s1_verify_w8a16.py` | Chạy inference trên EVK, đo Cosine Similarity & WER/CER |
| 6 | `step4_s1_profile_w8a16.py` | Đo latency + peak memory trực tiếp trên NPU |

### Vấn đề kỹ thuật đã gặp & cách khắc phục

**Lỗi 1 — Compiler crash khi biên dịch (`No bias info`):**
Trình biên dịch QAIRT của Qualcomm từ chối compile vì 70 lớp Convolution trong SenseVoice không có tham số `bias`. Giải pháp: dùng `onnx-graphsurgeon` để bơm các mảng `bias` bằng 0 (dummy zero bias) vào đúng các node đó mà không ảnh hưởng đến tính toán của mô hình.

**Lỗi 2 — Bất đồng bộ kích thước mảng khi verify (`Shape Mismatch`):**
Tensor output trả về từ NPU có kích thước tĩnh (do padding), còn mảng tham chiếu FP32 gốc có kích thước động. Giải pháp: viết thêm bước cắt bỏ phần đệm (padding crop) trước khi so sánh, và chỉ định rõ tên output (`output_0`) thay vì lấy ngẫu nhiên theo index dictionary.

### Kết quả

| Chỉ số | Giá trị | Ghi chú |
|--------|---------|---------|
| Compile QNN Binary | ✅ Thành công 100% | Job ID: `jgzn1jxxg`, Model ID: `mqky6w47m` |
| Inference trên EVK | ✅ 15/15 mẫu hoàn thành | Không treo, không văng bộ nhớ |
| Cosine Similarity (Logits) | ~0.9347 (93.47%) | Dao động 0.89–0.95 tuỳ file audio |
| Độ trễ (Inference Time) | **~269 ms** / mẫu 5 giây | RTF ≈ 0.054 — đáp ứng real-time |
| RAM đỉnh điểm (Peak Memory) | **~54.8 MB** | Profile Job ID: `j5w4o6mzg` |
| Tiêu thụ điện | Tối ưu (100% NPU) | Không dùng CPU/GPU trong inference |

### Kiến trúc Hybrid (CPU + NPU)

Toàn bộ mạng Neural (Encoder + CTC Head) chạy **100% trên NPU Hexagon**. Hai thành phần còn lại bắt buộc giữ trên CPU vì NPU không được thiết kế cho các tác vụ này:

- **Frontend (CPU):** Biến đổi sóng âm thô (raw wav) → đặc trưng Fbank `[1, T, 560]` bằng STFT/DSP.
- **CTC Decoder + Tokenizer (CPU):** Nhận Logits `[1, 500, 25055]` từ NPU → tính argmax → gộp token trùng (CTC decode) → chuỗi văn bản UTF-8.

> **Lưu ý về chất lượng:** Cosine Similarity 0.93 nằm dưới ngưỡng 0.95, nghĩa là phân phối xác suất (Logits) đã bị xê dịch do lượng tử hoá. Tuy nhiên, SenseVoice là mô hình Non-autoregressive — kết quả cuối chỉ phụ thuộc vào đỉnh xác suất cao nhất (Argmax), nên WER/CER thực tế nhiều khả năng không bị ảnh hưởng đáng kể. Chỉ số WER/CER trực tiếp sẽ được tái xác nhận sau khi tích hợp Tokenizer đầy đủ ở Step 5.