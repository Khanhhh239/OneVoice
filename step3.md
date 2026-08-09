# Step 3 — TTS (Text-to-Speech / Synthesis)

**Status (2026-08-09):** Xác nhận đề bài YÊU CẦU TTS (xem §0). Research mạnh xong dựa trên tra cứu thật (không suy đoán) — kiến trúc **ĐỀ XUẤT** (chưa code-test, giống trạng thái step0.md ban đầu): Supertonic (Vi+Ko+En) + MeloTTS-ZH (Zh). Bước tiếp theo: code-test thật giống Step 1/2.

---

## 0. Xác nhận yêu cầu

Đề bài (mục "The challenge"): *"On-device ML combining **speech recognition, translation, and synthesis** — no cloud dependency, optimized for speed and accuracy."* — **"synthesis" = TTS, là 1 trong 3 module bắt buộc** cùng ASR (Step 1) và MT (Step 2). Không phải tuỳ chọn.

---

## Part A — Drop-in cho Technical Proposal §4.2 "Module-by-Module Design"

| Module | Model / Framework | Size (est.) | Latency Target | Key Technique |
|---|---|---|---|---|
| TTS — Vietnamese / Korean / English | Supertonic (ONNX-native) | ~99M params | RTF ~0.3× đo trên Raspberry Pi CPU (chưa có số Snapdragon thật) | Multilingual VITS-family, chạy CPU-only, không cần GPU/NPU |
| TTS — Mandarin | MeloTTS-ZH (Qualcomm AI Hub) | Encoder 8.3M + Flow 20.1M + Decoder 14.5M + BERT-wrapper 152M (~745MB tổng, có thể quantize) | **Đo THẬT trên Snapdragon 8 Elite Gen 5**: Encoder 23.8ms + Decoder 42.5ms + Flow 71.2ms ≈ **~138ms/thành phần** | NPU-accelerated (HTP), pre-quantized sẵn trên AI Hub |

**Đây là module DUY NHẤT trong cả 3 step (0/1/2/3) có số latency đo thật trên phần cứng Snapdragon** — không phải ước lượng từ GPU dev machine như mọi module khác. Lý do: MeloTTS-ZH được Qualcomm tự profile và đăng công khai trên AI Hub.

**⚠️ Cần xác nhận trước khi chốt:** license Supertonic — sample code MIT nhưng **model dùng OpenRAIL-M** (Responsible AI License, có điều khoản hạn chế sử dụng — cần đọc kỹ điều khoản trước khi dùng cho thiết bị thương mại hoá sau này). MeloTTS license cần xác nhận lại (thường ghi nhận MIT nhưng chưa tự kiểm tra file LICENSE gốc).

---

## Part B — Phân tích đầy đủ, có số liệu chọn/loại từng candidate (Vietnamese)

### 1. Bài toán: không có model nào phủ đủ cả 4 ngôn ngữ

Giống hệt tình huống Step 1 (SenseVoice không có tiếng Việt) — TTS cũng vậy: **chưa tìm được model nào hỗ trợ tốt cả Vi+En+Zh+Ko trong 1 checkpoint duy nhất.**

| Candidate | Vi | En | Zh | Ko | Tham số | Có trên AI Hub? |
|---|---|---|---|---|---|---|
| **Supertonic 3** | ✅ | ✅ | ❌ | ✅ | ~99M | Không |
| **MeloTTS** (đầy đủ, MyShell.ai) | ❌ | ✅ | ✅ | ✅ | ~195M/ngôn ngữ | Chỉ bản **ZH** được Qualcomm tự optimize |
| **Kokoro-82M** | ❌ (đang phát triển, chưa dùng được) | ✅ | ✅ | ✅ | 82M | Không |
| **CosyVoice2-0.5B** | Chưa xác nhận | ✅ | ✅ | ✅ | 0.5B | Không |
| **Piper** (lựa chọn cũ trong kế hoạch) | ⚠️ có nhưng chất lượng thấp ("x_low" tier) | ✅ (tốt nhất nhóm) | ❌ | ❌ (cộng đồng chưa có giọng khả dụng) | ~15-60MB/giọng | Không |
| **VietTTS / VieNeu-TTS** (Việt chuyên biệt) | ✅ (tốt nhất cho Việt) | ❌ | ❌ | ❌ | Nhỏ, ONNX, CPU real-time | Không |

### 2. Quyết định: Supertonic (Vi+Ko+En) + MeloTTS-ZH (Zh)

**✅ CHỌN: Supertonic cho Việt + Hàn + Anh.** Lý do bằng số:
- **Duy nhất trong các candidate multilingual hỗ trợ cả Việt lẫn Hàn** — 2 ngôn ngữ khó nhất để tìm TTS tốt (Kokoro chưa có Việt; MeloTTS chưa có Việt; Piper không có Hàn khả dụng).
- Gộp được 3/4 ngôn ngữ vào 1 model duy nhất (99M tham số) — giảm đáng kể số model phải quản lý/quantize so với phương án 4 model riêng.
- Thiết kế sẵn cho edge: ONNX-native, chạy CPU-only, RTF 0.3× đo trên Raspberry Pi (phần cứng yếu hơn Snapdragon nhiều) — cho tín hiệu tốt về khả năng chạy real-time trên Snapdragon.

**✅ CHỌN: MeloTTS-ZH cho tiếng Trung.** Lý do: **model TTS duy nhất có số latency đo THẬT trên chip Snapdragon** (không phải suy đoán) — tự động thoả mãn yêu cầu "Technical Performance" (50% điểm) với bằng chứng cứng, và đã pre-quantize sẵn cho NPU qua AI Hub (giảm rủi ro kỹ thuật khi convert).

**❌ LOẠI: Piper (dù là lựa chọn cũ trong kế hoạch trước đây).** Lý do cụ thể: tiếng Hàn "cộng đồng chưa sản xuất được giọng khả dụng" (theo nguồn tra cứu) — loại hẳn khỏi vai trò multilingual; tiếng Việt chỉ có giọng chất lượng thấp nhất ("x_low" tier vẫn thắng các lựa chọn khác nghĩa là ngay cả lựa chọn TỐT NHẤT của Piper cho Việt vẫn ở mức thấp). Piper vẫn là lựa chọn tốt nếu CHỈ cần tiếng Anh, nhưng không đáp ứng đa ngôn ngữ ở đây.

**❌ LOẠI: Kokoro-82M.** Tiếng Việt "đang phát triển, chưa dùng được" — loại ngay vì tiếng Việt là ngôn ngữ trung tâm bắt buộc (mọi cặp dịch đều có Vi ở 1 đầu).

**❌ LOẠI: CosyVoice2-0.5B.** Không có bằng chứng hỗ trợ tiếng Việt; kích thước 0.5B lớn hơn Supertonic (~99M) 5 lần cho cùng nhóm ngôn ngữ En/Zh/Ko mà Supertonic (cộng MeloTTS cho Zh) đã phủ được.

**Backup nếu chất lượng tiếng Việt của Supertonic không đạt khi test thật**: VietTTS hoặc VieNeu-TTS (chuyên biệt tiếng Việt, VieNeu-TTS quảng cáo "on-device, real-time CPU inference, ONNX Runtime, int8 by default") — chấp nhận thêm 1 model riêng cho Việt nếu chất lượng đa ngôn ngữ của Supertonic không đủ tốt.

### 3. Rủi ro & điều cần làm rõ trước khi chốt

| Rủi ro | Ghi chú |
|---|---|
| License Supertonic model = OpenRAIL-M | Cần đọc điều khoản Responsible AI License trước khi cam kết dùng, đặc biệt nếu tính đến thương mại hoá sau cuộc thi (đề bài có chấm điểm Business Viability) |
| Chưa có số latency Supertonic trên Snapdragon thật | Chỉ có số Raspberry Pi CPU (RTF 0.3×) — cần tự đo trên Snapdragon/QCS6490 qua AI Hub remote-profile, giống việc còn thiếu ở Step 1/2 |
| MeloTTS-ZH khá nặng (152M cho riêng BERT-wrapper, ~745MB tổng chưa quantize) | Cần qua bước quantize INT8 trước khi deploy — đã có sẵn trên AI Hub nên bước này Qualcomm đã làm sẵn phần lớn |
| Chưa code-test chất lượng giọng nói thật (MOS, độ tự nhiên) cho cả 2 model | Đây là bước tiếp theo bắt buộc — nghe thử mẫu audio thật trước khi chốt, không chỉ dựa vào thông số kỹ thuật |

### 4. Kế hoạch test tiếp theo (theo đúng chuẩn Step 1/2)

1. Cài Supertonic + MeloTTS-ZH, sinh audio mẫu cho câu tiếng Việt/Hàn/Anh/Trung thật (tái sử dụng câu FLORES-200 đã có từ Step 2).
2. Đo RTF trên máy dev (GPU/CPU) làm baseline so sánh — ghi rõ đây KHÔNG phải số Snapdragon thật, giống cách đã làm ở Step 1/2.
3. Đánh giá chất lượng giọng nói: nghe thử trực tiếp + (nếu có thời gian) chạy MOS thử nghiệm nội bộ hoặc dùng metric tự động (UTMOS hoặc tương đương).
4. Nếu chất lượng tiếng Việt của Supertonic không đạt, thử thay bằng VietTTS/VieNeu-TTS riêng cho nhánh Việt, giữ Supertonic cho Hàn+Anh.
5. Dùng `qai-hub` để profile MeloTTS-ZH + (nếu chuyển đổi được) Supertonic trên Snapdragon/QCS6490 thật — đóng khoảng trống "chưa có số phần cứng thật" đã nêu ở Step 1/2/3.

---

**Document version:** 2026-08-09 — research-based (tra cứu thật, chưa code-test), Part A khớp format §4.2 Technical Proposal chính thức.
**Bước tiếp theo:** Code-test Supertonic + MeloTTS-ZH thật (sinh audio, đo RTF, nghe chất lượng) giống quy trình đã làm ở Step 1/2.
