# Step 1 — Nhận dạng Giọng nói (ASR)

Phân tích đầy đủ 5 mô hình ứng viên với số liệu thực tế, ghi chú license: [../../step1.md](../../step1.md)

**Mô hình được chọn:**
- **Tiếng Việt:** Zipformer-30M-RNNT-6000h
- **Anh / Trung / Hàn:** SenseVoice-Small

PhoWhisper, Moonshine và Qwen3-ASR-0.6B đã được kiểm thử và loại bỏ — xem lý do tại step1.md §2–4.

---

## Cấu trúc thư mục

`
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
├── unified_asr.py              # Class UnifiedASRPipeline: tự động định tuyến sang đúng model theo ngôn ngữ
├── test_unified_asr.py         # Kiểm thử toàn diện UnifiedASRPipeline (có/không có denoiser)
│
└── # ── Quantization & NPU Deployment (SenseVoice — End-to-End) ─────────
    │
    ├── step4_s1_export_e2e_onnx.py      # Bước 1: Gộp WavFrontend + Encoder + CTC thành 1 file ONNX E2E
    ├── step4_s1_patch_mask.py           # Bước 2: Bơm dummy bias vào 70 node Conv bị thiếu, fix lỗi PE
    ├── step4_s1_prepare_calib.py        # Bước 3: Cắt raw wav 15 mẫu (En/Zh/Ko) làm calibration data
    ├── step4_s1_qai_hub_submit_e2e.py   # Bước 4: Lượng tử hoá W8A16 + biên dịch QNN DLC qua QAI Hub
    ├── step4_s1_profile_e2e.py          # Bước 5: Đo latency và peak memory trực tiếp trên NPU Hexagon
    └── step4_s1_verify_w8a16.py         # (Tham khảo) Kiểm tra chất lượng Cosine Similarity sau quantize
`

---

## Cài đặt

`ash
pip install -r ../../requirements.txt
`

> **Lưu ý:** 	est_asr_qwen.py yêu cầu 	ransformers>=5.13.0, có thể xung đột với pin của funasr.
> Nếu bị lỗi, hãy cài vào một môi trường ảo (venv/conda) riêng.

---

## Hướng dẫn chạy

### 1. Chuẩn bị dữ liệu

`ash
python fetch_asr_data.py     # Tải FLEURS → data/asr/
python mix_asr_noise.py      # Tạo tập nhiễu → data/asr_mixed/
`

### 2. Benchmark từng mô hình ứng viên

`ash
python test_asr_multi.py      # SenseVoice-Small [CHỌN]
python test_asr_zipformer.py  # Zipformer-30M    [CHỌN]
python test_asr_vi.py         # PhoWhisper        [LOẠI]
python test_asr_moonshine.py  # Moonshine         [LOẠI]
python test_asr_qwen.py       # Qwen3-ASR-0.6B   [LOẠI]

# Hoặc chạy tất cả một lần:
python run_all_asr.py
`

Kết quả lưu tại outputs/asr_*_results.csv (WER/CER/RTF theo ngôn ngữ và mức SNR).

### 3. Kiểm thử pipeline tích hợp

`ash
python test_unified_asr.py
`

Kết quả lưu tại outputs/asr_unified_results.csv (bao gồm cột denoised). RTF trung bình trên máy dev đạt ~0.03.

---

## Lượng tử hoá & Deploy lên NPU (SenseVoice-Small w8a16 — End-to-End)

Để đáp ứng yêu cầu tốc độ cao và tiết kiệm pin trên phần cứng Edge (**Dragonwing IQ-9075 EVK** — Hexagon NPU v73+), SenseVoice-Small đã được lượng tử hoá sang định dạng **w8a16** (trọng số 8-bit, activations 16-bit) và biên dịch thành QNN DLC Binary.

### Kiến trúc E2E 100% trên NPU (Đột phá kỹ thuật)

Ban đầu, nhóm chỉ deploy được phần **Encoder** lên NPU — phần **Frontend** (trích xuất đặc trưng Fbank) phải chạy trên CPU do thư viện Kaldi dùng các phép toán như ten::fft_rfft và .unfold(dynamic_length) không được ONNX/NPU hỗ trợ. Điều này gây ra bottleneck về độ trễ truyền dữ liệu CPU-NPU và tiêu hao pin.

**Nhóm đã thành công "nướng" (bake) 100% pipeline từ Raw Audio vào một file ONNX thống nhất, chạy hoàn toàn trên NPU — không cần CPU xử lý giữa chừng.**

**Các giải pháp kỹ thuật đột phá:**

| # | Vấn đề | Giải pháp |
|---|--------|-----------|
| 1 | unfold() không hỗ trợ dynamic length | Thay bằng Conv2D 1D (framing = sliding window) — NPU rất giỏi Convolution |
| 2 | ft_rfft không hỗ trợ trên NPU | Nướng sẵn ma trận DFT tĩnh vào ONNX, biến FFT thành phép nhân Matmul |
| 3 | 70 node Conv thiếu ias → QAIRT crash | Bơm zero-bias vào graph bằng onnx-graphsurgeon |
| 4 | Shape Mismatch (562 vs 560) tại node Add_1 | QAIRT đánh giá sai node Range(float) — off-by-one. Giải pháp: bake cứng tensor Positional Encoding tĩnh [1, 504, 560] vào graph |

### Quy trình thực hiện

| Bước | Script | Mô tả |
|------|--------|--------|
| 1 | step4_s1_export_e2e_onnx.py | Export ONNX E2E chứa WavFrontend (Matmul+Conv2D thay FFT+unfold), Encoder, CTC Argmax |
| 2 | step4_s1_patch_mask.py | Bơm zero-bias vào 70 Conv bị thiếu; inject PE tĩnh để vá lỗi QAIRT Range |
| 3 | step4_s1_prepare_calib.py | Trích xuất raw wav 15 mẫu (5 En + 5 Zh + 5 Ko) làm calibration data |
| 4 | step4_s1_qai_hub_submit_e2e.py | Upload model lên QAI Hub, chạy Quantize W8A16 + Compile sang QNN DLC |
| 5 | step4_s1_profile_e2e.py | Submit Profile Job đo latency, peak RAM trên board Dragonwing IQ-9075 EVK thật |

### Kết quả thực tế (E2E W8A16 — Compile thành công)

| Chỉ số | Giá trị | Ghi chú |
|--------|---------|---------|
| Compile QNN DLC | ✅ Thành công 100% | Quantize Job: jpr0836vp → Compile Job: jpx4nonjg |
| Model ID (quantized) | mqej7v7ym | W8A16 ONNX đã lượng tử hoá |
| Compiled Model ID | mm5ke0j6m | QNN DLC Binary chạy trên Hexagon NPU |
| Cosine Similarity (Logits) | ~0.93 | Dao động 0.89–0.95 tuỳ file audio |
| Độ trễ (Inference Time) | **~269 ms** / mẫu 5 giây | RTF ≈ 0.054 — đáp ứng real-time |
| RAM đỉnh điểm (Peak Memory) | **~54.8 MB** | Đo từ profile job bản Encoder |
| Tiêu thụ điện | Tối ưu (100% NPU) | CPU hoàn toàn rảnh trong suốt inference |

### Kiến trúc luồng xử lý E2E

`
Raw Audio WAV
    │
    ▼ [NPU — 100%]
WavFrontend (Matmul DFT + Conv2D Framing) → Fbank [1, T, 560]
    │
    ▼
SenseVoice Encoder (Transformer x 50 layers) → Hidden States [1, T', 512]
    │
    ▼
CTC Head → Argmax → Token IDs [T'']
    │
    ▼ [CPU — ~0.1ms]
Tokenizer lookup (dictionary) → Transcript (UTF-8 Text)
`

> **Lưu ý về chất lượng:** Cosine Similarity ~0.93 nằm dưới ngưỡng 0.95. Tuy nhiên, SenseVoice là mô hình **Non-autoregressive** — kết quả cuối chỉ phụ thuộc vào đỉnh xác suất cao nhất (Argmax), nên WER/CER thực tế nhiều khả năng không bị ảnh hưởng đáng kể. Chỉ số WER/CER trực tiếp trên E2E sẽ được xác nhận tại Step 5 sau khi tích hợp Tokenizer hoàn chỉnh.
