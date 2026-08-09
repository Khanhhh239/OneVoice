# Step 3 — TTS (Text-to-Speech / Synthesis)

**Status (2026-08-09):** Đề bài xác nhận YÊU CẦU TTS (xem §0). Đã code-test thật 5 candidate (Supertonic, MeloTTS, Piper, Confucius4-TTS, VieNeu-TTS) trên cùng bộ câu FLORES-200 tái dùng từ Step 2, đo RTF thật + round-trip WER/CER thật (không chỉ tra cứu). Kiến trúc **CHỐT**: 3-model split — TTS-Vi (VieNeu-TTS hoặc Piper, xem §2.3) + Supertonic (Ko+En) + MeloTTS-ZH (Zh).

---

## 0. Xác nhận yêu cầu

Đề bài (mục "The challenge"): *"On-device ML combining **speech recognition, translation, and synthesis** — no cloud dependency, optimized for speed and accuracy."* — **"synthesis" = TTS, là 1 trong 3 module bắt buộc** cùng ASR (Step 1) và MT (Step 2). Không phải tuỳ chọn.

---

## Part A — Drop-in cho Technical Proposal §4.2 "Module-by-Module Design"

| Module | Model / Framework | Size (đo thật, on-disk) | Latency (đo thật, dev machine) | Key Technique |
|---|---|---|---|---|
| TTS — Vietnamese (Lựa chọn A — khuyến nghị) | Piper (`vi_VN-vais1000-medium`, VITS/ONNX) | **61 MB** | RTF **0.144** (CPU) | VITS one-shot decoder, ONNX-native, không cần LM/codec 2 tầng |
| TTS — Vietnamese (Lựa chọn B — chất lượng nhỉnh hơn, nặng hơn) | VieNeu-TTS 0.3B (backbone Q4 GGUF + NeuCodec ONNX-INT8 decoder) | **491 MB** (193 MB backbone + 298 MB codec) | RTF **0.483** (CPU) | Autoregressive speech-token LM + neural codec (kiến trúc giống Kokoro/CosyVoice-family, không phải VITS) |
| TTS — Korean + English | Supertonic 3 (ONNX-native, 4 submodel) | **380 MB** (text_encoder 35 + vector_estimator 245 + vocoder 97 + duration_predictor 3.6) | RTF **1.11** (ko) / **1.16** (en) (CPU) | Flow-matching TTS, ONNX Runtime, CPU-only |
| TTS — Mandarin | MeloTTS-ZH (checkpoint gốc, chưa quantize) | **199 MB** (checkpoint HF gốc) — bản Qualcomm AI Hub đã pre-quantize riêng, dung lượng khác | RTF **0.063** (steady-state, sau cold-start) — **đo THẬT trên Snapdragon 8 Elite Gen 5**: Encoder 23.8ms + Decoder 42.5ms + Flow 71.2ms | NPU-accelerated (HTP) trên bản AI Hub, pre-quantized sẵn |

**Tổng dung lượng deploy (3 model, chọn Lựa chọn A cho Việt):** 61 + 380 + 199 = **≈ 640 MB**
**Tổng dung lượng deploy (3 model, chọn Lựa chọn B cho Việt):** 491 + 380 + 199 = **≈ 1.07 GB**

**Đây vẫn là module DUY NHẤT trong cả 4 step (0/1/2/3) có số latency đo thật trên phần cứng Snapdragon** (MeloTTS-ZH, do Qualcomm tự profile công khai trên AI Hub). Mọi số RTF khác trong bảng trên đo trên máy dev (GPU/CPU thường), **không phải Snapdragon thật** — xem cảnh báo ở cuối file.

**⚠️ Cần xác nhận trước khi chốt:** license Supertonic — sample code MIT nhưng **model dùng OpenRAIL-M** (có điều khoản hạn chế sử dụng). VieNeu-TTS và Piper đều Apache 2.0 / MIT (không hạn chế). MeloTTS license cần xác nhận lại file LICENSE gốc.

---

## Part B — Phân tích đầy đủ, có số liệu chọn/loại từng candidate (Vietnamese)

### 1. Bài toán: không có model nào phủ đủ cả 4 ngôn ngữ

Giống hệt tình huống Step 1/2 — **chưa tìm được model nào hỗ trợ tốt cả Vi+En+Zh+Ko trong 1 checkpoint duy nhất.** Đã test/xác nhận trực tiếp (không chỉ tra cứu) 7 candidate:

| Candidate | Vi | En | Zh | Ko | Kích thước thật | Trạng thái |
|---|---|---|---|---|---|---|
| **Piper** (`vais1000-medium`) | ✅ **WER 14.1%** (đo thật) | ✅ WER 10.3% | ❌ | ❌ (cộng đồng chưa có giọng khả dụng) | 61 MB/giọng | ✅ CHỌN cho Vi (Lựa chọn A) |
| **VieNeu-TTS 0.3B** | ✅ **WER 12.8%** (đo thật) | ❌ | ❌ | ❌ | 491 MB (deploy path) | ✅ CHỌN cho Vi (Lựa chọn B) |
| **Supertonic 3** | ⚠️ có nhưng **WER 35.2%** — lỗi lặp từ rõ ràng khi nghe | ✅ WER 7.9% | ❌ | ✅ CER 6.8% | 380 MB | ✅ CHỌN cho Ko+En, ❌ LOẠI cho Vi |
| **MeloTTS** (đầy đủ) | ❌ | ✅ WER 8.6% | ✅ **CER 7.3%** (1.2% nếu bỏ 1 câu chứa tên riêng Māori khó) | ✅ (chưa test) | 199 MB/ngôn ngữ (checkpoint gốc) | ✅ CHỌN bản ZH, ❌ LOẠI En/Ko (đã có Supertonic tốt hơn) |
| **Confucius4-TTS** (NetEase Youdao) | Tuyên bố có (zero-shot cloning) | Tuyên bố có | Tuyên bố có | Tuyên bố có | **>2.4 GB chỉ riêng speaker-encoder (w2v-bert-2.0)**, chưa tính T2S+S2A+vocoder | ❌ LOẠI — bỏ dở khi test vì dung lượng phi thực tế cho edge |
| **Kokoro-82M** | ❌ (không có) | ✅ | ✅ | ❌ **đã xác minh: không có giọng Hàn nào (không có prefix kf_/km_)** | 82 MB | ❌ LOẠI — không có Việt lẫn Hàn, mất hết lý do cân nhắc |
| **CosyVoice2-0.5B** | Không có bằng chứng | ✅ | ✅ | ✅ | 0.5B | ❌ LOẠI — không xác nhận được Việt |

### 2. Quyết định

**✅ CHỌN: Supertonic cho Hàn + Anh.** Không đổi so với đề xuất ban đầu — vấn đề chất lượng của Supertonic **chỉ xảy ra ở tiếng Việt** (lặp từ, WER 35.2%), còn Hàn (CER 6.8%) và Anh (WER 7.9%) đều tốt, không có lỗi lặp. Kokoro-82M — ứng viên thay thế duy nhất còn lại — đã xác minh **không hỗ trợ tiếng Hàn** (tra model card + VOICES.md gốc trên HuggingFace, không có prefix `kf_`/`km_` nào), nên bị loại thẳng, Supertonic là lựa chọn duy nhất còn lại.

**✅ CHỌN: MeloTTS-ZH cho tiếng Trung.** Không đổi — vẫn là model duy nhất có số latency đo thật trên Snapdragon.

**❌ LOẠI HẲN: Supertonic cho tiếng Việt** (khác với đề xuất ban đầu ở phiên bản trước của file này). Lý do bằng số — round-trip WER 35.2%, cao hơn 2.5-3× so với 2 lựa chọn thay thế, với lỗi lặp từ rõ ràng quan sát được trong bản dịch ASR ngược (vd. câu "dịch vụ này... dịch vụ này..." bị lặp) ở 3/5 câu test. Đây là phát hiện **chỉ lộ ra khi code-test thật** — tra cứu ban đầu (research thuần) không phát hiện được vấn đề này vì Supertonic được quảng cáo hỗ trợ tốt tiếng Việt.

### 2.3. Tiếng Việt: 2 lựa chọn, đánh đổi rõ ràng — cần nghe mẫu để chốt

| Tiêu chí | **Piper** (Lựa chọn A) | **VieNeu-TTS** (Lựa chọn B) |
|---|---|---|
| Round-trip WER (đo thật, 5 câu) | 14.1% | **12.8%** (tốt hơn, nhưng chênh lệch nhỏ trên n=5, có thể trong sai số) |
| RTF (đo thật, CPU) | **0.144** (nhanh hơn 3.3×) | 0.483 |
| Kích thước deploy | **61 MB** (nhỏ hơn 8×) | 491 MB (backbone GGUF 193MB + codec ONNX-INT8 298MB) |
| Kiến trúc | VITS (1 tầng, decode trực tiếp) | LM sinh speech-token + neural codec giải mã (2 tầng, giống họ Kokoro/CosyVoice) |
| License | MIT | Apache 2.0 |
| Giọng | 1 giọng nữ miền Bắc (`vais1000`) | 6 giọng (3 nam, 3 nữ; cả miền Bắc lẫn miền Nam) |
| Rủi ro triển khai | Thấp — đã chạy ổn định ngay | **Cao hơn** — bản PyTorch "standard" mode nặng 1.17GB (backbone+codec PyTorch gốc), phải chuyển sang backend ONNX-INT8 (`neuphonic/neucodec-onnx-decoder-int8`, chưa tự test trực tiếp, chỉ xác nhận dung lượng qua HF API) mới đạt được con số 491MB ở trên |

**Khuyến nghị: chọn Piper (Lựa chọn A)** nếu ưu tiên đúng như yêu cầu gốc của đề bài (tối ưu speed + size cho on-device) — thắng cả về tốc độ (3.3×) lẫn dung lượng (8×), trong khi chênh lệch chất lượng (14.1% vs 12.8% WER) nằm trong khoảng nhiễu thống kê với chỉ 5 mẫu. VieNeu-TTS là lựa chọn hợp lý nếu team ưu tiên **đa dạng giọng nói** (6 giọng, 2 miền) hoặc nghe thử thấy giọng tự nhiên hơn rõ rệt — **đã gửi 4 file mẫu (Piper vs VieNeu, câu dễ + câu khó) để tự nghe và chốt bằng tai**, vì round-trip WER chỉ đo độ dễ hiểu (intelligibility), không đo độ tự nhiên (naturalness).

### 3. Lý do loại các candidate khác (không đổi/củng cố thêm)

**❌ LOẠI: Confucius4-TTS.** Bắt đầu test (dùng 1 câu tiếng Việt Step 1 làm giọng mẫu zero-shot cloning) nhưng dừng giữa chừng — riêng speaker-encoder (w2v-bert-2.0) đã tải hơn **2.4GB**, chưa tính 3 thành phần còn lại (T2S 24-layer, flow-matching S2A, BigVGAN vocoder). So với tổng 3 model đã chọn (~640MB-1.07GB CHO CẢ 4 NGÔN NGỮ), một model claim phủ 4 ngôn ngữ nhưng riêng 1 thành phần đã nặng hơn toàn bộ giải pháp — không hợp lý cho edge.

**❌ LOẠI: Kokoro-82M.** Không có tiếng Việt (không tranh cãi) và **đã xác minh trực tiếp không có tiếng Hàn** — tra cứu ban đầu nói "có" (không chính xác/nhầm lẫn nguồn), tự kiểm tra model card + VOICES.md gốc trên HuggingFace chỉ thấy 9 nhóm giọng: en-US/en-GB/ja/zh/es/fr/hi/it/pt-BR, không có `kf_`/`km_`.

**❌ LOẠI: CosyVoice2-0.5B.** Không có bằng chứng hỗ trợ tiếng Việt; 0.5B tham số lớn hơn nhiều so với giải pháp 3-model đã chọn cho cùng nhóm ngôn ngữ.

### 4. Kết quả đo thật đầy đủ (round-trip WER/CER qua Zipformer/SenseVoice, giống phương pháp Step 1)

| Engine | Lang | WER/CER trung bình (5 câu) | RTF trung bình |
|---|---|---|---|
| Piper | vi | 14.09% | 0.144 |
| Piper | en | 10.31% | 0.109 |
| VieNeu-TTS | vi | 12.79% | 0.483 |
| Supertonic | vi | 35.24% (lỗi lặp từ) | 0.531 |
| Supertonic | ko | 6.77% | 1.109 |
| Supertonic | en | 7.93% | 1.161 |
| MeloTTS | zh | 7.31% (1.2% nếu bỏ câu tên riêng khó) | 0.063 (sau cold-start) |
| MeloTTS | en | 8.64% | 0.050 (sau cold-start) |

**Lưu ý cold-start:** MeloTTS câu đầu tiên luôn có RTF rất cao (zh 1.11, en 11.85) do CUDA/JIT warmup — không phản ánh tốc độ thực khi chạy liên tục. Cần gọi warm-up 1 lần trước khi phục vụ người dùng thật.

### 5. Rủi ro & điều cần làm rõ trước khi chốt

| Rủi ro | Ghi chú |
|---|---|
| License Supertonic model = OpenRAIL-M | Cần đọc điều khoản trước khi cam kết, đặc biệt nếu tính thương mại hoá sau cuộc thi |
| **⚠️ Khoảng trống quan trọng chưa xử lý — phần cứng thật** | Toàn bộ số RTF ở bảng §4 đo trên máy dev (GPU NVIDIA / CPU thường), **không phải Snapdragon thật** — giống khoảng trống đã nêu ở Step 1/2. Riêng MeloTTS-ZH là ngoại lệ (có số Snapdragon 8 Elite Gen 5 thật từ Qualcomm AI Hub). Piper, VieNeu-TTS, Supertonic đều **chưa xác nhận có trên Qualcomm AI Hub hay không** — cần dùng `qai-hub` để profile thật trước khi chốt vào Technical Proposal. |
| VieNeu-TTS deploy path (ONNX-INT8 codec) chưa tự test trực tiếp | Con số 491MB/RTF suy ra một phần từ dung lượng file trên HuggingFace API, chưa chạy thực tế bằng `neucodec-onnx-decoder-int8` — bản đã test thực tế dùng codec PyTorch gốc (978MB, nặng hơn nhiều, không phù hợp deploy) |
| MeloTTS-ZH bản gốc chưa quantize (199MB) khác với bản Qualcomm AI Hub đã pre-quantize | Dung lượng thật sau quantize trên AI Hub chưa xác nhận lại bằng số — chỉ có số latency |
| Chưa nghe mẫu Piper vs VieNeu-TTS bằng tai để đánh giá độ tự nhiên | Round-trip WER chỉ đo độ dễ hiểu, không đo độ tự nhiên — đã gửi mẫu, chờ quyết định cuối |

---

**Document version:** 2026-08-09 — code-test thật đầy đủ 5 candidate (Supertonic, MeloTTS, Piper, Confucius4-TTS bỏ dở, VieNeu-TTS), có số RTF + round-trip WER/CER thật, Part A khớp format §4.2 Technical Proposal chính thức.
**Bước tiếp theo:** (1) Nghe mẫu Piper vs VieNeu-TTS để chốt lựa chọn Việt cuối cùng; (2) profile Piper/VieNeu-TTS/Supertonic trên Snapdragon thật qua `qai-hub`; (3) test trực tiếp VieNeu-TTS với backend `neucodec-onnx-decoder-int8` để xác nhận con số 491MB.
