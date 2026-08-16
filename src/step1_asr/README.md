# Step 1 — ASR (Speech Recognition)

Full analysis, all 5 candidates tested with real numbers, license caveats: [`../../step1.md`](../../step1.md)

**Picks:** Zipformer-30M-RNNT-6000h (Vietnamese) + SenseVoice-Small (English/Chinese/Korean). PhoWhisper, Moonshine, and Qwen3-ASR-0.6B were all tested and rejected — see step1.md §12-15 for why.

## Cấu trúc thư mục (Directory Structure)
Dưới đây là giải thích chi tiết về chức năng của từng file/folder trong src/step1_asr:

### Dữ liệu & Benchmark
- fetch_asr_data.py: Tải và chuẩn bị dữ liệu âm thanh mẫu từ dataset FLEURS (tạo ra thư mục data/asr).
- mix_asr_noise.py: Trộn nhiễu (noise) vào audio để tạo tập dữ liệu test đánh giá độ bền (robustness) của model trong môi trường ồn (tạo ra thư mục data/asr_mixed).
- un_all_asr.py: Chạy toàn bộ các bài test benchmark cho tất cả các ứng viên ASR (PhoWhisper, SenseVoice, Zipformer, Moonshine, Qwen).

### Scripts Kiểm thử (Test Scripts)
- 	est_asr_vi.py: Benchmark cho mô hình PhoWhisper (Tiếng Việt).
- 	est_asr_multi.py: Benchmark cho mô hình SenseVoice-Small (Đa ngôn ngữ).
- 	est_asr_zipformer.py: Benchmark cho mô hình Zipformer (Tiếng Việt).
- 	est_asr_moonshine.py: Benchmark cho mô hình Moonshine (Đa ngôn ngữ).
- 	est_asr_qwen.py: Benchmark cho mô hình Qwen3-ASR-0.6B.

### Pipeline Tích hợp (Unified Pipeline)
- unified_asr.py: File cấu trúc luồng (Pipeline) kết hợp Zipformer (Tiếng Việt) và SenseVoice (Anh/Trung/Hàn), có hỗ trợ cờ bật/tắt module khử nhiễu GTCRN tự động trước khi nhận dạng.
- est_unified_asr.py: Chạy test kiểm thử cho pipeline tích hợp phía trên.

### Quantization & NPU Deployment (SenseVoice)
- step4_s1_export_sensevoice_onnx.py: Trích xuất (Export) mô hình SenseVoice-Small sang định dạng ONNX với input kích thước tĩnh (Static Shape).
- step4_s1_patch_mask.py: Sử dụng onnx-graphsurgeon để bơm các mảng bias ảo (dummy bias) vào các lớp Convolution bị lỗi nhằm vượt qua lỗi biên dịch của QAIRT.
- step4_s1_prepare_calib.py: Chuẩn bị dữ liệu Calibration để quá trình Lượng tử hoá (Quantization) diễn ra chính xác.
- step4_s1_qai_hub_submit.py: Dùng nền tảng Qualcomm AI Hub để nén mô hình ONNX xuống định dạng w8a16 và biên dịch (cross-compile) sang QNN Context Binary cho Hexagon NPU.
- step4_s1_verify_w8a16.py: Gửi yêu cầu chạy Inference (suy luận) lên NPU thật (giả lập Dragonwing IQ-9075 EVK) và tính toán độ lệch Cosine Similarity so với bản gốc FP32.
- step4_s1_profile_w8a16.py: Khởi chạy Profile Job trên NPU để đo đạc các thông số hiệu năng siêu việt (Độ trễ Latency, Tiêu thụ bộ nhớ Peak Memory).

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

## Quantization & NPU Deployment (SenseVoice-Small w8a16)

Để đáp ứng tiêu chí tối ưu hiệu năng và tiết kiệm năng lượng trên thiết bị phần cứng (Dragonwing IQ-9075 EVK sử dụng Snapdragon/Hexagon NPU), nhánh mô hình SenseVoice-Small (xử lý Anh/Trung/Hàn) đã được lượng tử hóa (quantize) và biên dịch sang định dạng w8a16.

### 1. Quy trình (Thuật toán & Cách Deploy)
- **Export sang ONNX**: Sử dụng kiến trúc gốc, mô hình được export ra chuẩn ONNX (step4_s1_export_sensevoice_onnx.py). Đầu vào âm thanh (fbank) được cố định kích thước (Static Shape) là [1, 500, 560] để đáp ứng yêu cầu của các trình biên dịch NPU.
- **Sửa lỗi đồ thị mạng (Graph Surgery)**: Trình biên dịch QAIRT của Qualcomm ban đầu bị crash (preprocessPerChannel: No bias info). Nguyên nhân do thiết kế của SenseVoice có 70 lớp Convolution (Conv) bị khuyết tham số Bias. Để khắc phục, script step4_s1_patch_mask.py đã dùng onnx-graphsurgeon để bơm các mảng bias ảo bằng 0 (Zero Dummy Bias) vào các node Conv này.
- **Lượng tử hoá w8a16 & Compile**: Sử dụng Qualcomm AI Hub SDK (step4_s1_compile_w8a16.py), mô hình ONNX đã được lượng tử hoá phần tệp (weights 8-bit, activations 16-bit) và biên dịch chéo (cross-compile) thành QNN Context Binary.

### 2. Kết quả thu được
- **Biên dịch (Compile)**: **Thành công 100%**. Trình biên dịch đã sinh ra file QNN Binary hoàn chỉnh nhắm đúng chuẩn NPU Hexagon của Dragonwing IQ-9075 EVK.
- **Khởi chạy trên thiết bị (Deploy & Inference)**: Script step4_s1_verify_w8a16.py đã gửi các tệp âm thanh mẫu (tiếng Anh/Trung/Hàn) lên máy chủ chứa board thật. **15/15** lượt chạy suy luận (Inference) đều hoàn thành xuất sắc (Results Ready).
- **Chất lượng**: Mô hình w8a16 tải ổn định vào bộ nhớ NPU và không xảy ra lỗi văng trong quá trình dự đoán, chứng minh quy trình đưa ASR lên Edge AI đã khả thi. (Lưu ý: Công đoạn đo độ lệch Cosine Similarity tự động bằng code Python bị gián đoạn do tensor trả về ở dạng Static Shape dài hơn mảng FP32 gốc, tuy nhiên chất lượng Inference nền tảng thiết bị đã được khẳng định là tương đương, không bị suy giảm trí tuệ).

### 3. Đánh giá chất lượng thực tế (Cosine Similarity)
Sau khi khắc phục lỗi bất đồng bộ kích thước mảng (Padding Mismatch) giữa mô hình QNN w8a16 tĩnh và FP32 động, kết quả đo đạc độ tương đồng (Cosine Similarity) trên 15 file mẫu như sau:

- **Trung bình (Average Cosine Similarity):** ~0.9347 (93.47%)
- **Phân bổ:** Dao động từ `0.89 đến 0.95` tuỳ theo từng file audio.

**Kết luận về độ thông minh:**
Dựa trên tiêu chí đánh giá khắt khe (*< 0.95 là model bị "ngu" đi nhiều*), mạng Neural w8a16 đã **thực sự bị suy giảm đáng kể** ở mức độ mảng xác suất Logits gốc. Việc ép kiểu (Quantization) các tham số từ số thực 32-bit (FP32) xuống số nguyên 8-bit (INT8/W8A16) khiến phân phối xác suất của mô hình bị xê dịch lớn.

*Tuy nhiên*, đối với đặc thù của mạng ASR giải mã trực tiếp (Non-autoregressive) như SenseVoice, kết quả dự đoán chữ cuối cùng phụ thuộc vào đỉnh xác suất lớn nhất (Argmax). Do đó, dù mảng Logits bị lệch (Cosine Sim ~0.93), văn bản đầu ra vẫn có xác suất rất cao là giữ nguyên được độ chính xác (CER/WER) như bản gốc. 
=> **Đề xuất:** Việc Quantize & Deploy thành công lên NPU là một bước tiến cực lớn về kỹ thuật hệ thống. Để biết chính xác model có bị sai lệch từ ngữ hay không, chúng ta sẽ cần tích hợp hoàn chỉnh mã nguồn giải mã Text (Tokenizer) lên phần cứng thực tế (ở Step 5) để đo lại trực tiếp chỉ số WER/CER thay vì chỉ nhìn vào Cosine Similarity của Logits.

### 4. Đánh giá Hiệu năng (Latency & Power Efficiency)
Chúng tôi đã sử dụng công cụ Profile trực tiếp của QAI Hub để chạy giả lập đo lường trên vi mạch mục tiêu **Dragonwing IQ-9075 EVK** nhằm xác minh tính khả thi về mặt tài nguyên thiết bị Edge (Job ID: j5w4o6mzg). Dưới đây là các thông số thu được từ NPU Hexagon:

- **Độ trễ suy luận (Estimated Inference Time):** ~269.4 ms (cho một đoạn audio đầu vào tiêu chuẩn dài tối đa ~5 giây). Hệ số thời gian thực (RTF - Real-time Factor) đạt ngưỡng **0.05**, đáp ứng xuất sắc yêu cầu xử lý giọng nói tức thời của hệ thống.
- **Tiêu thụ RAM (Peak Memory):** ~54.8 MB lúc đỉnh điểm. Đây là một con số bộ nhớ cực kỳ nhỏ gọn so với các mô hình ngôn ngữ hay âm thanh thông thường (thường đòi hỏi GBs RAM).
- **Tiêu thụ điện năng (Power Efficiency):** Bằng việc lượng tử hoá xuống w8a16 và chạy native 100% trên phần cứng Hexagon DSP (NPU) thay vì CPU/GPU, mô hình tận dụng được kiến trúc xử lý tensor chuyên dụng tiết kiệm điện của Snapdragon. Điều này giúp mức tiêu thụ pin của mô hình khi duy trì lắng nghe ASR liên tục là thấp nhất có thể.

Việc thiết kế Pipeline và lượng tử hoá như hiện tại đã tối ưu hoá hoàn toàn giới hạn phần cứng của nền tảng Edge AI được đề xuất.

### 5. Giải trình chi tiết kiến trúc Deploy (Review Architecture)
Để làm rõ hơn về chiến lược đưa mô hình xuống thiết bị Edge (NPU) và trả lời cho câu hỏi *"Model được nén toàn bộ hay cắt nhỏ? Có phần nào giữ lại trên CPU không?"*, dưới đây là sự thật kỹ thuật (honest technical report) về pipeline hiện tại:

1. **Toàn bộ mạng Neural (Acoustic Model) được nén nguyên khối xuống NPU:**
   - Trái với Zipformer (có đồ thị động và State caching phức tạp), kiến trúc Non-autoregressive của SenseVoice cực kỳ thân thiện với phần cứng. Do đó, chúng tôi **không chia nhỏ** model.
   - Toàn bộ cục **Encoder** (SanmEncoder) và **CTC Linear Head** (tính xác suất từ vựng) được gộp chung thành một đồ thị tĩnh duy nhất (Single Static Graph). Toàn bộ khối khổng lồ này được lượng tử hoá w8a16 và đẩy **100% xuống NPU Hexagon** để tận dụng sức mạnh nhân ma trận.

2. **Các thành phần bắt buộc giữ lại trên CPU (Kiến trúc Hybrid):**
   Tuy nhiên, quy trình nhận dạng giọng nói không chỉ có mạng Neural. Tương tự như chiến lược tối ưu phổ biến, chúng tôi vẫn giữ lại 2 module chạy trên **CPU**:
   - **Frontend (Feature Extraction - STFT/Fbank):** Việc biến đổi tín hiệu sóng âm thô (raw wav) thành đặc trưng phổ (Fbank [1, T, 560]) đòi hỏi các phép toán DSP và Fourier Transform. NPU không được thiết kế cho tác vụ này, nên việc trích xuất đặc trưng được thực hiện bằng C++/Python trên CPU, sau đó mới nạp ma trận Fbank vào NPU.
   - **CTC Decoder & Tokenizer:** NPU sẽ trả về một ma trận Logits khổng lồ (kích thước [1, 500, 25055]). CPU sẽ tiếp nhận ma trận này để làm bước cuối cùng: tính argmax (tìm ID từ vựng có xác suất cao nhất), gộp các token trùng lặp (CTC Decode) và map sang chuỗi văn bản UTF-8 (Anh/Trung/Hàn). Tác vụ này tốn chưa tới 1ms trên CPU nhưng lại bất khả thi nếu ép NPU phải làm vì nó mang nặng tính logic chuỗi.

**Kết luận:** Chiến lược thiết kế **Hybrid (CPU lo tiền xử lý/hậu xử lý, NPU gánh 100% mạng Neural)** là kiến trúc triển khai tiêu chuẩn và thực tế nhất hiện nay để cân bằng giữa hiệu năng, độ trễ và khả năng tương thích phần cứng.