# Step 4 — Hardware & Quantization (Deploy target + model compression)

**Status (2026-08-09):** Chốt phần cứng demo: **Rubik Pi 3 (Qualcomm QCS6490)**, có phương án dự phòng. Đo thật dung lượng on-disk của TẤT CẢ model đã chọn ở Step 0-3 (không phải ước lượng) — phát hiện quan trọng: **SenseVoice-Small (893MB) và NLLB-600M (2.3GB) chưa hề được quantize**, dù step1.md/step2.md từng ghi "~250MB (int8-quantizable)" như một ước lượng chưa thực hiện. Tổng dung lượng hiện tại (chưa nén): **~3.86GB**. Mục tiêu sau khi quantize: **~1.3GB** — có kế hoạch cụ thể theo từng model ở Part B.

---

## Part A — Drop-in cho Technical Proposal §5 "Hardware & Device Concept"

### Bảng so sánh phần cứng

| Platform | NPU | RAM | Giá | Portability | Trên Qualcomm AI Hub? |
|---|---|---|---|---|---|
| **Rubik Pi 3 (QCS6490) — CHỌN** | 12 TOPS | 8GB LPDDR4x (nguồn: retailer, chưa xác nhận official) | ~$179 ($159 early-bird, nguồn thứ cấp) | ⚠️ Cần nguồn USB-C PD 3.0 12V/3A (36W) — KHÔNG có pin sẵn | ✅ Có, chính thức |
| Snapdragon 8 Elite Gen 5 (phone) — dự phòng | ~80 TOPS (marketing claim tới ~100, chưa xác nhận số chính xác) | 12-16GB (tuỳ máy) | $1000+ | ✅ Pin sẵn, cầm tay thật sự, zero rủi ro nguồn điện | Suy luận từ tooling docs, chưa xác nhận trực tiếp trong danh sách device AI Hub |
| Thundercomm TurboX C8550 (QCS8550) | Chưa công bố TOPS | Chưa công bố | Chưa công bố (phải liên hệ sales) | Chưa rõ | ✅ Có, nhưng gắn nhãn **"(Proxy)"** — Qualcomm tự ghi "metrics sẽ khác trên thiết bị thật" |
| QCS8300 | — | — | — | — | ❌ Không thấy trong danh sách AI Hub — loại khỏi cân nhắc |

### Power budget (ước tính, chưa đo thật)

| Thành phần | Ghi chú |
|---|---|
| Rubik Pi 3 board | Yêu cầu 12V/3A = 36W đầu vào (USB-C PD 3.0) |
| Cần đo thật | Chưa có số công suất thực đo khi chạy pipeline đầy đủ — action item cho Step 5 |

**⚠️ Việc CHƯA xác nhận rõ trước khi chốt:** Rubik Pi 3 không có pin — để thực sự "portable" theo đúng yêu cầu đề bài ("any portable device format"), cần thêm 1 power bank hỗ trợ PD 3.0 12V (đa số power bank phổ thông chỉ ra 5V/9V, KHÔNG đủ) hoặc mạch buck-boost riêng. Đây là rủi ro tích hợp thật, chưa có trong ngân sách/kế hoạch trước đây.

---

## Part B — Phân tích đầy đủ (Vietnamese)

### 1. Vì sao chọn Rubik Pi 3, không phải phương án khác

**✅ CHỌN: Rubik Pi 3 (QCS6490) làm thiết bị demo chính.** Lý do:
- Duy nhất trong 3 phương án có đầy đủ thông tin công khai (giá, NPU TOPS, RAM) — TurboX C8550 phải liên hệ sales mới biết giá, rủi ro cho timeline cuộc thi.
- **Có trên Qualcomm AI Hub chính thức** (không gắn nhãn Proxy) — nghĩa là số latency đo qua `qai-hub` sẽ là số thật trên đúng chip, không phải suy diễn từ thiết bị thay thế như TurboX C8550.
- Giá hợp lý (~$179) so với phương án dùng điện thoại flagship (~$1000+).
- Form-factor RPi-HAT tương thích — dễ gắn thêm mic array (ReSpeaker 4-mic) đã có trong kế hoạch từ trước.

**⚠️ Rủi ro mới phát hiện (chưa từng ghi nhận trước đây):** Rubik Pi 3 cần nguồn 12V/3A qua USB-C PD 3.0, KHÔNG có pin sẵn. Power bank thường (5V/9V) sẽ KHÔNG chạy được board này. Đây là rủi ro trực tiếp tới tiêu chí "portable device" của đề bài — phải xác nhận mua power bank PD 3.0 12V tương thích, hoặc build mạch nguồn riêng, trước khi cam kết ngày demo.

**Phương án dự phòng: Snapdragon 8 Elite Gen 5 phone.** Nếu rủi ro tích hợp nguồn điện của Rubik Pi 3 không giải quyết kịp trước deadline, dùng điện thoại flagship Snapdragon — có pin sẵn, cầm tay thật, NPU mạnh hơn nhiều (~80 TOPS vs 12 TOPS) nên dư sức chạy cả 4 module cùng lúc mà không cần quantize sâu như Rubik Pi 3. Đánh đổi: giá cao hơn 5-6 lần, và **chưa xác nhận trực tiếp có trong danh sách device AI Hub** (chỉ suy luận từ tooling docs) — cần tự kiểm tra qua `qai-hub` trước khi cam kết.

**❌ LOẠI: QCS8300.** Không xuất hiện trong danh sách device chính thức của Qualcomm AI Hub (đã kiểm tra trực tiếp trang docs) — không tìm được dev kit hay giá công khai. Loại khỏi cân nhắc cho tới khi có nguồn xác nhận tốt hơn.

**⚠️ Cân nhắc thêm nhưng chưa đủ dữ liệu: Qualcomm RB3 Gen2/RB5** — cùng chip QCS6490, do chính Qualcomm làm dev kit robotics, nhưng chưa xác nhận giá — nếu rẻ hơn hoặc có support tốt hơn Rubik Pi 3 thì đáng cân nhắc lại, cần thêm 1 vòng tra cứu giá trước khi hoàn toàn loại bỏ.

### 2. Tổng dung lượng model thật (đo on-disk, không ước lượng) — TRƯỚC khi quantize

| Step | Model | Format hiện tại | Size thật |
|---|---|---|---|
| 0 | Silero VAD | ONNX | 1.3 MB |
| 0 | GTCRN | PyTorch checkpoint | 0.6 MB |
| 1 | Zipformer-30M (Vi) | ONNX **int8 đã có sẵn** | **29.3 MB** (encoder 27 + decoder 1.3 + joiner 1.0) |
| 1 | SenseVoice-Small (En/Zh/Ko) | PyTorch **fp32, CHƯA quantize** | **893 MB** |
| 2 | NLLB-200-distilled-600M | safetensors **fp32, CHƯA quantize** | **2,300 MB** |
| 3 | Piper (Vi) | ONNX (đã gọn sẵn) | 61 MB |
| 3 | Supertonic (Ko+En) | ONNX **fp32, CHƯA quantize** | 380 MB |
| 3 | MeloTTS-ZH (Zh) | PyTorch checkpoint gốc **fp32, CHƯA quantize** (bản AI Hub đã tự quantize riêng, dung lượng khác chưa xác nhận) | 199 MB |
| | **TỔNG (chưa quantize)** | | **≈ 3,864 MB ≈ 3.86 GB** |

**Phát hiện quan trọng:** step1.md Part A từng ghi SenseVoice-Small "~250MB (int8-quantizable)" — đây chỉ là **con số ước lượng cho bản ĐÃ quantize, chưa từng thực sự chạy quantize**. Bản thật đang dùng để test là fp32, nặng gấp 3.6×. Đây là khoảng trống giống hệt kiểu lỗi đã bắt được ở Step 3 (VieNeu-TTS deploy path) — số liệu "ước lượng" bị nhầm thành "đã làm".

### 3. Kế hoạch quantize cụ thể theo từng model

| Model | Công cụ khuyến nghị | Size ước tính sau quantize | Lý do chọn công cụ |
|---|---|---|---|
| Zipformer-30M | *(đã xong)* | 29.3 MB | Tác giả model đã export sẵn bản int8 qua sherpa-onnx — dùng thẳng |
| SenseVoice-Small | ONNX Runtime dynamic/static int8 (hoặc export tool của funasr) | ~223 MB (ước tính theo tỷ lệ fp32→int8 = 1/4) | Model non-autoregressive, kiến trúc đơn giản — int8 dynamic quantization rủi ro thấp |
| NLLB-600M | **CTranslate2 int8** (`ct2-transformers-converter --quantization int8`) | ~600 MB | **KHÔNG dùng QNN w4a16** — recipe đó của Qualcomm tối ưu cho LLM decoder-only (Llama-style), chưa có bằng chứng công khai áp dụng tốt cho kiến trúc encoder-decoder như NLLB. CTranslate2 int8 là đường đã kiểm chứng rộng rãi cho MT seq2seq, rủi ro thấp hơn |
| Supertonic | ONNX Runtime int8 (quantize riêng từng submodel: text_encoder/vector_estimator/vocoder/duration_predictor) | ~100-190 MB (ước tính) | 4 submodel ONNX độc lập — quantize từng cái, cần tự đo WER sau quantize để không lặp lại lỗi "chưa test thật" |
| MeloTTS-ZH | **Dùng thẳng bản Qualcomm AI Hub đã pre-quantize** (không tự quantize) | Chưa xác nhận số MB — chỉ có số latency | Qualcomm đã làm và profile thật trên Snapdragon 8 Elite Gen 5 — tự làm lại vừa tốn công vừa khó đạt chất lượng ngang bản chính chủ |
| Piper | Không cần (đã đủ nhỏ, 61MB) | 61 MB | ROI thấp — ưu tiên quantize NLLB/SenseVoice trước vì 2 model đó chiếm 82% tổng dung lượng hiện tại |

**Tổng dung lượng mục tiêu sau khi quantize: ≈ 1.3 GB** (1.3 + 0.6 + 29.3 + 223 + 600 + 61 + 190 + 199 ≈ 1,304 MB) — giảm **~66%** so với 3.86GB hiện tại, nằm thoải mái trong RAM 8GB của Rubik Pi 3 kể cả khi tính thêm buffer cho OS + audio pipeline runtime.

### 4. Rủi ro & việc cần làm rõ trước khi chốt hẳn vào Technical Proposal

| Rủi ro | Ghi chú |
|---|---|
| RAM Rubik Pi 3 = 8GB chỉ xác nhận qua nguồn thứ cấp (retailer) | Chưa tìm được specsheet chính thức ghi rõ có bản 4GB hay không — cần hỏi trực tiếp Thundercomm/nhà phân phối |
| Nguồn điện 12V/3A PD 3.0, không có pin | Cần mua/test power bank tương thích PD 3.0 12V TRƯỚC ngày demo — nếu không thiết bị không thực sự "portable" |
| Giá TurboX C8550 (QCS8550) chưa công khai | Không đưa vào kế hoạch chính vì rủi ro timeline (phải liên hệ sales, không rõ lead time) |
| Snapdragon 8 Elite Gen 5 chưa xác nhận trực tiếp trong danh sách AI Hub | Chỉ suy luận từ tooling docs (LiteRT blog) — cần tự kiểm tra qua `qai-hub` trước khi chọn làm phương án dự phòng chính thức |
| Chưa đo power budget thật khi chạy full pipeline | Chỉ có specsheet nguồn vào (36W), chưa đo công suất tiêu thụ thực tế lúc inference — cần đo sau khi có board thật |
| SenseVoice-Small và NLLB-600M CHƯA quantize | Đây là việc làm tiếp theo bắt buộc trước khi có thể chạy trên Rubik Pi 3 thực tế trong RAM 8GB — ước tính size sau quantize ở §3 chưa được tự chạy/verify, chỉ tính theo tỷ lệ lý thuyết fp32→int8 |
| Chưa profile bất kỳ model nào (trừ MeloTTS-ZH) trên Snapdragon thật qua `qai-hub` | Khoảng trống đã nêu xuyên suốt Step 1/2/3 — Step 4 xác nhận lại: đây vẫn là việc quan trọng nhất còn lại trước khi nộp Technical Proposal |

---

**Document version:** 2026-08-09 — nghiên cứu thật (web search + kiểm tra trực tiếp Qualcomm AI Hub device docs), đo thật on-disk size toàn bộ model Step 0-3, có kế hoạch quantize cụ thể theo từng model. Phần cứng CHỐT: Rubik Pi 3 (QCS6490), dự phòng Snapdragon 8 Elite Gen 5 phone.
**Bước tiếp theo:** (1) Mua/test power bank PD 3.0 12V cho Rubik Pi 3; (2) chạy quantize thật cho SenseVoice-Small (ONNX int8) và NLLB-600M (CTranslate2 int8), đo lại WER/BLEU sau quantize để xác nhận không tụt chất lượng; (3) dùng `qai-hub` profile toàn bộ pipeline trên Rubik Pi 3 thật.
