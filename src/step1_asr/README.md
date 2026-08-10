# Step 1 — ASR (Speech Recognition)

Full analysis, all 5 candidates tested with real numbers, license caveats: [`../../step1.md`](../../step1.md)

**Picks:** Zipformer-30M-RNNT-6000h (Vietnamese) + SenseVoice-Small (English/Chinese/Korean). PhoWhisper, Moonshine, and Qwen3-ASR-0.6B were all tested and rejected — see step1.md §12-15 for why.

## Setup

```bash
pip install -r ../../requirements.txt
```

`test_asr_qwen.py` needs `transformers>=5.13.0`, which can conflict with funasr's own pin — install that one in an isolated venv/conda env if a plain `pip install -U transformers` breaks `test_asr_vi.py` / `test_asr_multi.py`.

## Run

```bash
python fetch_asr_data.py      # -> data/asr/<lang>/*.wav + manifest.json (FLEURS-based)
python mix_asr_noise.py       # -> data/asr_mixed/<lang>/*_snrN.wav (SNR-robustness test set)

python test_asr_vi.py         # PhoWhisper (Vi) -- rejected candidate, kept for comparison
python test_asr_multi.py      # SenseVoice-Small (En/Zh/Ko) -- current pick
python test_asr_zipformer.py  # Zipformer-30M (Vi) -- current pick
python test_asr_moonshine.py  # Moonshine (Vi/En/Zh/Ko) -- rejected candidate
python test_asr_qwen.py       # Qwen3-ASR-0.6B -- rejected (10x slower than SenseVoice)

# or run the current-pick pair + all alt candidates in order:
python run_all_asr.py
```

Results -> `outputs/asr_*_results.csv` (WER/CER/RTF per language, per SNR level).

## Unified ASR Pipeline (Cải tiến mới)

Dựa trên kết luận giữ kiến trúc **Split** (Zipformer cho Tiếng Việt + SenseVoice cho Anh/Trung/Hàn), mã nguồn đã được tái cấu trúc thành một Pipeline thống nhất duy nhất để dễ dàng tích hợp vào Step 5 (End-to-end pipeline) sau này.

- `unified_asr.py`: Đóng gói bộ 2 mô hình (Zipformer và SenseVoice) vào class `UnifiedASRPipeline`. Class này nhận input `(audio_path, lang)` và tự động định tuyến sang model phù hợp. **Đặc biệt:** Hỗ trợ cờ `UnifiedASRPipeline(use_denoiser=True)` để tự động tích hợp mô hình khử nhiễu **GTCRN** (từ Step 0) làm sạch âm thanh trước khi nhận dạng, giúp tăng cường khả năng xử lý trên dữ liệu ồn ào.
- `test_unified_asr.py`: Script kiểm thử toàn diện class `UnifiedASRPipeline` trên toàn bộ tập dữ liệu. Tự động tính toán WER/CER và RTF (độ trễ) cho 4 ngôn ngữ. Đối với dữ liệu nhiễu (`asr_mixed`), script sẽ tự động test cả 2 kịch bản (có dùng và không dùng Denoiser) để so sánh đối chiếu.

### Hướng dẫn chạy Unified Test

```bash
cd src/step1_asr

# Đảm bảo đã tải và chuẩn bị dữ liệu (bao gồm cả dữ liệu mix nhiễu)
python fetch_asr_data.py
python mix_asr_noise.py

# Chạy test đánh giá Pipeline thống nhất (bao gồm test khử nhiễu)
python test_unified_asr.py
```

Kết quả tổng hợp sẽ được lưu tại `outputs/asr_unified_results.csv` (có thêm cột `denoised` báo hiệu file đó đã được làm sạch tiếng ồn hay chưa). Tốc độ trung bình (RTF) của pipeline thống nhất đạt khoảng `~0.03` trên máy tính cá nhân.
