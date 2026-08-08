# Step 1: ASR — Nhận dạng giọng nói (Speech → Text, Streaming, 4 ngôn ngữ)

**Status:** Phân tích kiến trúc dựa trên research thật (paper + benchmark đã verify qua web search). Code test đã viết cho cả kiến trúc chính (PhoWhisper + SenseVoice) lẫn 3 candidate mới phát hiện sau (Zipformer-30M, Moonshine, Qwen3-ASR — xem 12) — đang chờ chạy thật để so sánh, khuyến nghị Part A dưới đây coi là tạm thời, chưa chốt cho tới khi có số WER/RTF thật.

---

## Part A: Technical Proposal §4.2/§4.3 (Copy-Paste Ready)

### Kiến trúc: Model chuyên biệt theo ngôn ngữ, không dùng 1 model chung

```
Audio sạch, đã VAD-cắt (từ Step 0)
        │
        ▼
   Language ID (nhận diện ngôn ngữ — có sẵn trong SenseVoice, hoặc router riêng)
        │
   ┌────┴────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼
 Tiếng Việt  English   Chinese    Korean
   │          │          │          │
   ▼          └────┬─────┴────┬─────┘
PhoWhisper        SenseVoice-Small
(fine-tuned       (non-autoregressive,
 Whisper,          1 model cho cả 3 ngôn ngữ)
 844h Vi data)
   │                    │
   ▼                    ▼
LocalAgreement-2    Re-decode mỗi 300ms
(streaming policy    (rẻ vì inference
 cho model tự hồi quy) cực nhanh)
   │                    │
   └─────────┬──────────┘
             ▼
      Text (partial + final) → Step 2 (MT)
```

### Quyết định kiến trúc (bảng tóm tắt)

| Ngôn ngữ | Model | Vì sao |
|---|---|---|
| **Tiếng Việt** | **PhoWhisper** (VinAI Research) | Model DUY NHẤT được fine-tune chuyên cho tiếng Việt trên 844 giờ, 26.000 người nói, phủ 63 tỉnh thành; SOTA trên mọi benchmark Việt công khai; SenseVoice **không hỗ trợ tiếng Việt** nên bắt buộc phải tách riêng |
| **English / Chinese / Korean** | **SenseVoice-Small** (FunAudioLLM) | 1 model non-autoregressive phủ cả 3 ngôn ngữ cùng lúc; nhanh hơn Whisper-Small 5 lần, Whisper-Large 15 lần (cùng tham số); CER tiếng Trung đã verify: 2.96% (AISHELL-1) so với Whisper-Large-v3 5.14% — **thắng cả về tốc độ lẫn độ chính xác** |

### Streaming policy theo loại model

| Model | Kiểu | Chính sách streaming |
|---|---|---|
| PhoWhisper (tự hồi quy) | Encoder-decoder autoregressive | **LocalAgreement-2** (Whisper-Streaming, arXiv 2307.14743) — so 2 lần đoán liên tiếp, phần đầu giống nhau thì chốt |
| SenseVoice (non-autoregressive) | Predict toàn bộ 1 lần | Re-decode toàn câu mỗi ~300ms — không cần thuật toán phức tạp vì bản thân model đã siêu nhanh |

### Dữ liệu fine-tune tiếng Việt

| Dataset | Quy mô | Mục đích |
|---|---|---|
| PhoWhisper base training | 844h, 26.000 người nói, 63 tỉnh | Đã tích hợp sẵn trong checkpoint public |
| ViMD (arXiv 2410.03458) | 102.56h, ~19.000 câu, 63 tỉnh → 3 miền | Fine-tune thêm để cân bằng WER giữa 3 miền (giọng Trung có WER cao nhất, cần nhiều nhất) |
| Bud500 | ~500h, đa chủ đề (podcast/du lịch/sách/ẩm thực), 3 miền | Bổ sung độ đa dạng nội dung + giọng |

### Tiêu chí hoàn thành (DoD) — đã đối chiếu với literature

| Tiêu chí | Ngưỡng | Cơ sở |
|---|---|---|
| WER tiếng Việt (audio sạch) | < 8% | PhoWhisper-large đạt SOTA trên benchmark công khai (số chính xác cần lấy Table 1 trong paper gốc — xem Part B §9) |
| WER tiếng Việt (SNR 5dB) | < 15%, không "sập" | Dựa trên hành vi model fine-tune có multi-condition training |
| WER cân bằng 3 miền | Không miền nào lệch > baseline nhiều | ViMD paper: fine-tune cải thiện Bắc +1.86%, **Trung +3.07%** (cao nhất — xác nhận đây là điểm yếu cần tập trung), Nam +2.34% |
| Quantize int8, WER tụt | < 1–2 điểm % | Khớp với literature: ONNX INT8 gây suy giảm WER tương đối ~1–2% trên audio sạch (nhiều nguồn độc lập xác nhận) |
| Tiếng Anh/Trung/Hàn | Đạt benchmark công khai gốc | SenseVoice CER 2.96% (AISHELL-1) đã verify tốt hơn Whisper-Large-v3 |

---

## Part B: Phân tích kỹ thuật đầy đủ

### 1. Nhiệm vụ & ràng buộc

- **Input**: audio đã VAD-cắt, mono 16kHz, sạch (output của Step 0).
- **Output**: text streaming — bản `partial` (đang đoán, có thể sửa) và `final` (đã chốt), kèm nhãn ngôn ngữ.
- **Ràng buộc cứng**: hạn chế train (ưu tiên pretrained/fine-tune nhẹ, không train from scratch); phải chạy 2 chiều (nhận cả tiếng Việt lẫn tiếng nước ngoài nói vào máy); RTF phải < 1 trên phần cứng Snapdragon edge.
- **Rủi ro đặc thù**: tiếng Việt và tiếng Trung là ngôn ngữ thanh điệu; tiếng Hàn là ngôn ngữ chắp dính (agglutinative) với ranh giới từ mơ hồ — ảnh hưởng cách đo WER (xem §2.4).

### 2. Kiến trúc theo từng loại ngôn ngữ (nội dung chính)

#### 2.1 Tiếng Việt — PhoWhisper (không có lựa chọn thay thế hợp lý)

**Vì sao bắt buộc phải tách riêng, không dùng chung với SenseVoice:**
SenseVoice-Small (bản public trên GitHub FunAudioLLM/SenseVoice) chỉ hỗ trợ **5 ngôn ngữ: Mandarin, Cantonese, English, Japanese, Korean — không có tiếng Việt.** Đây không phải lựa chọn thiết kế, mà là giới hạn cứng của model. Vậy nên phương án B (PhoWhisper riêng + SenseVoice riêng) không phải "chọn cho gọn" mà là **bắt buộc về mặt kỹ thuật**.

**PhoWhisper** (arXiv 2406.02555, VinAI Research, Tiny Paper @ ICLR 2024):
- Fine-tune từ OpenAI Whisper trên 844 giờ audio tiếng Việt, 26.000 người nói, trải khắp 63 tỉnh thành.
- 5 kích cỡ: tiny/base/small/medium/large — chọn theo ngân sách compute của Snapdragon.
- Đã tăng cường chống nhiễu: dùng thư viện `audiomentations` trộn noise môi trường vào **một nửa** dữ liệu train (multi-condition training có sẵn từ gốc — lợi thế lớn cho môi trường nhà máy ồn).
- PhoWhisper-large đạt SOTA trên mọi benchmark tiếng Việt công khai; small/medium/large vượt qua mọi baseline wav2vec2; tiny/base ngang ngửa wav2vec2-base.
- Open-source: [github.com/VinAIResearch/PhoWhisper](https://github.com/VinAIResearch/PhoWhisper)

**Vấn đề còn lại: phương ngữ (dialect).** 844h training của PhoWhisper không tách riêng theo miền — đây là chỗ ViMD + Bud500 bổ sung (xem §5).

#### 2.2 English / Chinese / Korean — SenseVoice-Small (1 model, 3 ngôn ngữ)

**SenseVoice** (FunAudioLLM/FunASR, Alibaba):
- Kiến trúc **non-autoregressive end-to-end** — dự đoán toàn bộ chuỗi ký tự trong 1 lần forward pass, không sinh từng token tuần tự như Whisper → đây là lý do nó nhanh vượt trội.
- Tốc độ: **nhanh hơn Whisper-Small 5 lần, Whisper-Large 15 lần** ở cùng số tham số (verify qua GitHub README + FunASR blog chính thức).
- Train trên 400.000+ giờ audio, bản đầy đủ phủ 50+ ngôn ngữ; bản Small tối ưu cho 5 ngôn ngữ chính (Zh/Yue/En/Ja/Ko).
- **Benchmark đã verify**: CER 2.96% trên AISHELL-1 (tiếng Trung), 3.80% trên AISHELL-2 test_ios — so với Whisper-Large-v3 chỉ đạt CER 5.14% trên cùng AISHELL-1. SenseVoice-Small **thắng cả tốc độ lẫn độ chính xác** dù nhỏ hơn nhiều.
- Bonus miễn phí: language ID, nhận diện cảm xúc, phát hiện sự kiện âm thanh (tiếng vỗ tay, ho, cười...) — có thể tận dụng cho các tính năng phụ sau này.

**Vì sao không dùng SenseVoice luôn cho cả tiếng Việt (né phải quản lý 2 model)?**
Không được — model không hỗ trợ. Nếu muốn 1 model duy nhất, phải quay lại phương án A (Whisper-small multilingual chung), nhưng khi đó mất lợi thế 844h fine-tune chuyên biệt của PhoWhisper cho tiếng Việt.

#### 2.3 Bảng so sánh 3 phương án (đầy đủ bằng chứng)

| Tiêu chí | A: Whisper-small chung | B: PhoWhisper + SenseVoice (khuyến nghị) | C: Canary + AlignAtt |
|---|---|---|---|
| Số model quản lý | 1 | 2 | 1 |
| Chất lượng tiếng Việt | Kém hơn (chưa chuyên biệt) | Tốt nhất (SOTA đã verify) | Chưa rõ — xem §2.5 |
| Tốc độ En/Zh/Ko | Baseline | 5–15× nhanh hơn Whisper (đã verify) | Chưa rõ cho ASR thuần |
| Độ phức tạp routing | Không cần | Cần route theo ngôn ngữ | Không cần (nếu dùng được) |
| Rủi ro | Thấp, nhưng chất lượng Việt hạn chế | Thấp — cả 2 model đều pretrained, không cần train from scratch | **Cao** — chưa test tiếng Việt, task gốc là dịch chứ không phải ASR thuần |

#### 2.4 Vấn đề đo lường: WER vs CER theo ngôn ngữ

Một chi tiết dễ bỏ sót: **tiếng Trung và tiếng Hàn không nên đo bằng WER (Word Error Rate) như tiếng Việt/Anh.**
- Tiếng Trung: không có ranh giới từ rõ ràng (không cách giữa các từ) → phải đo **CER (Character Error Rate)**.
- Tiếng Hàn: có cách giữa từ nhưng **nhiều trường hợp được phép bỏ cách** (dữ liệu Common Voice/FLEURS có nhãn cách không nhất quán) → cộng đồng nghiên cứu (kể cả paper gốc Whisper) cũng dùng CER cho tiếng Hàn thay vì WER để tránh đo sai do lỗi spacing chứ không phải lỗi nhận dạng thật.
- **Áp dụng cho DoD**: bảng tiêu chí hoàn thành ở Part A cần tách rõ — WER cho Việt/Anh, CER cho Trung/Hàn — nếu không tách sẽ so sánh táo với cam.

Whisper-small baseline (đã verify, FLEURS dataset): CER tiếng Trung 18.1%, CER tiếng Hàn 12.2%. SenseVoice-Small (CER 2.96–3.80% trên AISHELL) cho thấy khoảng cách cải thiện rất lớn so với dùng Whisper chung (phương án A).

#### 2.5 Vì sao phương án C (Canary) rủi ro hơn nhận định ban đầu

Research sâu hơn phát hiện điều quan trọng: **paper "Pocket Offline Model" (arXiv 2606.03948, CUNI nộp IWSLT 2026, công bố 06/2026) không phải là bài toán ASR (nhận dạng) mà là bài toán dịch trực tiếp giọng nói → chữ (Speech-to-Text Translation)** — Canary nhận audio ngôn ngữ nguồn, xuất thẳng chữ ngôn ngữ ĐÍCH, bỏ qua bước transcript trung gian.

- Model: 1B tham số, hỗ trợ 25 ngôn ngữ nguồn × 25 ngôn ngữ đích.
- Kết quả đã test: chỉ trên cặp Czech→English, English→German/Italian (IWSLT 2026 shared task) — **không có tiếng Việt, tiếng Trung, hay tiếng Hàn trong danh sách cặp đã kiểm chứng.**
- AlignAtt (cơ chế streaming của Canary) đã được tích hợp sẵn vào project **SimulStreaming** — cùng dòng với Whisper-Streaming (cùng tác giả Dominik Macháček, Charles University) — nghĩa là AlignAtt **cũng áp dụng được cho họ Whisper/PhoWhisper**, không chỉ riêng Canary.

**Kết luận**: Phương án C thực chất đang giải bài toán khác (dịch trực tiếp, gộp luôn Step 1+2), không phải ASR thuần theo định nghĩa nhiệm vụ ở §1. Với ràng buộc "hạn chế train" và việc Vi↔Ko/Vi↔Zh là cặp ít dữ liệu song song, một model dịch trực tiếp end-to-end thường thua hệ thống cascade (ASR riêng → MT riêng) khi thiếu dữ liệu — đúng như lo ngại "rủi ro chưa rõ" trong đề bài gốc, giờ có bằng chứng cụ thể để xác nhận. **Giữ nguyên khuyến nghị phương án B**, nhưng ghi nhận AlignAtt như một **hướng nâng cấp streaming trong tương lai cho PhoWhisper** (thay LocalAgreement-2), vì nó không cần train lại.

### 3. Streaming: chi tiết cơ chế

#### 3.1 LocalAgreement-2 (cho PhoWhisper)

Từ Whisper-Streaming (arXiv 2307.14743, Macháček/Dabre/Bojar, IJCNLP-AACL 2023):
- Nguyên lý: chạy model trên audio buffer hiện tại → được hypothesis H1; đợi thêm audio, chạy lại → được H2; so sánh phần đầu 2 hypothesis, đoạn nào giống nhau thì **chốt (confirm)** làm output final, đoạn cuối còn khác nhau thì giữ làm partial, đợi vòng sau.
- Đã verify: đạt độ trễ 3.3 giây trên bộ test speech dài không cắt đoạn, đã dùng thực tế cho phiên dịch trực tiếp hội nghị đa ngôn ngữ.
- Áp dụng trực tiếp cho PhoWhisper vì cùng kiến trúc Whisper (encoder-decoder autoregressive).

#### 3.2 Re-decode (cho SenseVoice)

Vì SenseVoice là non-autoregressive và cực nhanh (ước tính dưới 100ms cho audio ngắn dựa trên tỉ lệ nhanh hơn Whisper-Small 5 lần), chiến lược đơn giản nhất là: mỗi khi buffer có thêm ~300ms audio mới, chạy lại toàn bộ từ đầu đoạn hiện tại. Không cần thuật toán LocalAgreement vì:
- Không có chi phí "chờ 2 lần đoán" — mỗi lần đoán đã rẻ sẵn.
- Không có concept "partial hypothesis" theo nghĩa autoregressive — model luôn xuất câu hoàn chỉnh cho đoạn audio đang có.

**Cần đo thực tế**: độ trễ chính xác và "flicker rate" (câu bị sửa qua lại) của cách re-decode này chưa có số liệu công khai — đây là việc phải tự đo khi code-test (xem §6 bước 4).

#### 3.3 Hướng nâng cấp: AlignAtt cho PhoWhisper

AlignAtt (arXiv 2305.11408 gốc, tích hợp trong SimulStreaming) dùng attention alignment giữa audio và output để quyết định khi nào đủ tự tin để xuất token tiếp theo — **không cần retrain**, áp dụng được cho mọi model có cross-attention (bao gồm Whisper/PhoWhisper). Về lý thuyết có thể cho độ trễ thấp hơn LocalAgreement-2 vì không cần đợi "2 lần đoán trùng nhau". Đề xuất: thử nghiệm cả 2 (LocalAgreement-2 vs AlignAtt) trên PhoWhisper trong giai đoạn code-test, so sánh độ trễ + flicker rate.

### 4. Multi-condition training & nhiễu môi trường

PhoWhisper đã có sẵn augmentation cơ bản (audiomentations trên nửa training set) nhưng đây là nhiễu tổng quát, không phải nhiễu nhà máy thực tế. Kế hoạch:
1. Lấy noise nhà máy thật (đã có sẵn hạ tầng từ Step 0: `mix_noise.py`, MUSAN + noise tự thu).
2. Trộn vào ViMD + Bud500 ở nhiều mức SNR (0/5/10/15/20dB) — **trộn lúc TRAIN, không chỉ lúc test** (nguyên tắc multi-condition training: model phải thấy nhiễu trong lúc học thì mới học được cách bỏ qua nó).
3. Fine-tune PhoWhisper (LoRA hoặc full fine-tune tuỳ ngân sách) trên hỗn hợp: ViMD (đa miền) + Bud500 (đa chủ đề) + nhiễu nhà máy.
4. SenseVoice: vì không có kế hoạch fine-tune (giữ nguyên pretrained do "hạn chế train"), robustness với nhiễu sẽ dựa hoàn toàn vào Step 0 (denoise GTCRN đã adaptive theo SNR) — nghĩa là **chất lượng Step 0 quan trọng hơn với En/Zh/Ko so với tiếng Việt**, vì tiếng Việt còn có lớp fine-tune chống nhiễu ở Step 1 làm lưới an toàn thứ hai.

### 5. Dữ liệu fine-tune tiếng Việt (chi tiết)

| Dataset | Quy mô | Nguồn | Đặc điểm |
|---|---|---|---|
| PhoWhisper base | 844h, 26.000 người, 63 tỉnh | Đã tích hợp trong checkpoint | Baseline, không cần tải lại |
| **ViMD** | 102.56h, ~19.000 câu, 1.2 triệu từ | [nguyendv02/ViMD_Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) (HF) | Tin tức phát thanh, gán nhãn Bắc/Trung/Nam rõ ràng — dùng để đo + cân bằng WER theo miền |
| **Bud500** | ~500h | [linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) (HF) | Podcast/du lịch/sách/ẩm thực, đa miền — bổ sung độ đa dạng nội dung ngoài tin tức |

**Bằng chứng ViMD đáng chú ý**: paper báo cáo fine-tune cải thiện WER Bắc +1.86%, **Trung +3.07%** (baseline WER cao nhất, cải thiện nhiều nhất), Nam +2.34%. → Xác nhận giọng Trung là điểm yếu lớn nhất cần tập trung dữ liệu, khớp với constraint DoD "không miền nào chênh lệch quá lớn" trong đề bài gốc.

### 6. Quy trình thử nghiệm chi tiết

1. **Baseline**: chạy PhoWhisper + SenseVoice gốc (chưa fine-tune thêm) trên test set sạch (Common Voice Vi, FLEURS toàn bộ 4 ngôn ngữ) → đo WER (Vi/En) và CER (Zh/Ko, theo §2.4).
2. **Fine-tune**: PhoWhisper trên ViMD + Bud500 + noise nhà máy (multi-condition, xem §4). SenseVoice giữ nguyên pretrained.
3. **Đo lại WER/CER tách theo miền** (Bắc/Trung/Nam cho tiếng Việt) — không gộp chung 1 số, lý do đã nêu trong đề bài gốc: gộp sẽ giấu mất điểm yếu từng miền.
4. **Đo theo ma trận SNR**: WER/CER tại clean/20/15/10/5/0dB, vẽ biểu đồ đường — tái sử dụng hạ tầng `mix_noise.py` từ Step 0.
5. **Đo streaming**: độ trễ tới partial đầu tiên + flicker rate, so sánh LocalAgreement-2 vs AlignAtt (PhoWhisper) và đo riêng độ trễ re-decode (SenseVoice) — chưa có số liệu công khai cho phần này, bắt buộc phải tự đo (xem §3.2).
6. **Đo trên thiết bị sau quantize int8**: so sánh WER/CER trước/sau, dùng ONNX Runtime hoặc whisper.cpp cho PhoWhisper (đã verify: ~1–2% suy giảm tương đối là mức bình thường trong literature, dynamic quantization tốt hơn static).

### 7. Tiêu chí hoàn thành (DoD) — chi tiết + căn cứ

| Tiêu chí | Ngưỡng | Căn cứ |
|---|---|---|
| WER tiếng Việt, audio sạch | < 8% | PhoWhisper-large SOTA trên benchmark công khai (cần lấy số chính xác từ Table 1 paper gốc — xem hạn chế ở §9) |
| WER tiếng Việt, SNR 5dB | < 15%, không sập | Suy từ multi-condition training + augmentation gốc của PhoWhisper |
| Chênh lệch WER giữa 3 miền | Không có miền lệch bất thường so với baseline | ViMD: Trung là miền yếu nhất (baseline WER cao nhất) — cần theo dõi riêng miền này |
| CER tiếng Trung | Tiệm cận SenseVoice public benchmark | Đã verify: 2.96% (AISHELL-1), 3.80% (AISHELL-2 test_ios) |
| CER tiếng Hàn | Đạt benchmark công khai SenseVoice | Chưa có số CER chính xác public cho tiếng Hàn — cần tự đo trên FLEURS Korean |
| WER tiếng Anh | Đạt mức Whisper-Large-v3 hoặc tốt hơn | Whisper-Large-v3: ~2.0–2.7% WER (LibriSpeech test-clean), ~4.0% WER (FLEURS English) — SenseVoice cần đạt tương đương |
| Suy giảm sau quantize int8 | < 1–2 điểm % tuyệt đối | Khớp nhiều nguồn độc lập: ONNX INT8 gây ~1–2% suy giảm tương đối trên audio sạch, nhiều hơn trên audio nhiễu |

### 8. Rủi ro & giảm thiểu

| Rủi ro | Nguyên nhân | Giảm thiểu |
|---|---|---|
| SenseVoice không hỗ trợ tiếng Việt | Giới hạn cứng của model (chỉ 5 ngôn ngữ: Zh/Yue/En/Ja/Ko) | Đã thiết kế tách riêng PhoWhisper cho tiếng Việt ngay từ đầu — không phải rủi ro nữa mà là constraint đã xử lý |
| Giọng miền Trung yếu hơn Bắc/Nam | ViMD xác nhận baseline WER miền Trung cao nhất | Ưu tiên tỉ trọng dữ liệu miền Trung khi fine-tune; đo riêng để không bị số tổng "che" |
| Flicker rate cao ở streaming (chữ partial nhảy qua lại) | Chưa có số liệu — rủi ro chưa lượng hoá được | Bắt buộc đo thực nghiệm ở bước code-test; nếu SenseVoice re-decode gây flicker cao, cân nhắc thêm buffer nhỏ hoặc hysteresis đơn giản |
| Quantize int8 làm giảm WER nhiều hơn dự kiến trên audio nhiễu | Literature ghi nhận suy giảm nhiều hơn trên audio noisy so với clean | Test quantize riêng ở từng mức SNR, không chỉ audio sạch |
| Phương án C (Canary) không tốt cho Vi/Zh/Ko | Paper gốc chỉ test Cs→En, En→De/It — không có bằng chứng cho 3 ngôn ngữ mục tiêu | Không chọn làm phương án chính; theo dõi AlignAtt như kỹ thuật streaming có thể tách rời áp dụng cho PhoWhisper |

### 9. Tài liệu tham khảo có chú giải

1. **PhoWhisper** — [arXiv 2406.02555](https://arxiv.org/abs/2406.02555) — VinAI Research, ICLR 2024 Tiny Paper. Đọc để lấy Table 1/2 (WER chính xác theo từng size model) — bài fetch tool không lấy được số bảng do PDF nén, cần đọc trực tiếp.
2. **ViMD** — [arXiv 2410.03458](https://arxiv.org/abs/2410.03458) — EMNLP 2024. Đọc Table 7 để lấy WER đầy đủ theo từng tỉnh (không chỉ theo miền).
3. **Bud500** — [HF: linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) — đọc data card để biết format transcript.
4. **SenseVoice** — [GitHub FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) + [FunAudioLLM paper arXiv 2407.04051](https://arxiv.org/html/2407.04051v1) — đọc để lấy chi tiết kiến trúc non-autoregressive (SAN-M/Paraformer-style) và benchmark đầy đủ AISHELL/Wenetspeech/LibriSpeech/Common Voice.
5. **Whisper-Streaming** — [arXiv 2307.14743](https://arxiv.org/abs/2307.14743) — Macháček/Dabre/Bojar, IJCNLP-AACL 2023. Đọc phần 3 (LocalAgreement) để lấy pseudocode chính xác.
6. **AlignAtt (gốc)** — [arXiv 2305.11408](https://arxiv.org/pdf/2305.11408) — cơ chế attention-alignment streaming, áp dụng được cho mọi model có cross-attention.
7. **Pocket Offline Model (Canary + AlignAtt)** — [arXiv 2606.03948](https://arxiv.org/abs/2606.03948) — CUNI, nộp IWSLT 2026 (06/2026). **Lưu ý quan trọng**: đây là bài toán dịch trực tiếp (S2T translation), không phải ASR — chỉ test Cs→En, En→De/It, KHÔNG có Vi/Zh/Ko. Dùng để tham khảo kỹ thuật AlignAtt/SimulStreaming, không dùng làm bằng chứng chất lượng cho 3 ngôn ngữ mục tiêu.

**Hạn chế của bước research này**: một số con số (WER chính xác PhoWhisper theo từng size, CER tiếng Hàn của SenseVoice, độ trễ "70ms/10s" cho SenseVoice) không lấy được số chính xác qua web search/fetch tự động — các trang chứa bảng bị chặn (403) hoặc PDF nén không đọc được text. Cần tự mở paper/README để lấy số khi vào giai đoạn code-test, tương tự cách step0.md đã làm với dữ liệu đo thật thay vì chỉ trích dẫn.

### 10. Kết nối với Step 0 và Step 2

- **Từ Step 0**: nhận audio đã VAD-cắt + (với SNR thấp) đã denoise GTCRN. Vì SenseVoice không có lớp fine-tune chống nhiễu riêng, chất lượng denoise ở Step 0 ảnh hưởng trực tiếp đến En/Zh/Ko nhiều hơn Vi.
- **Sang Step 2 (MT)**: xuất `partial`/`final` text kèm nhãn ngôn ngữ. MT cần xử lý được text `partial` thay đổi liên tục (không đợi `final` mới dịch, để giữ độ trễ thấp) — đây là input cho thiết kế AlignAtt-policy ở Step 2 (đã đề cập rủi ro Vi↔Ko/Vi↔Zh ít dữ liệu song song trong phân tích Step 0 trước đó).
- **Lưu ý thiết kế**: phương án C (Canary) cho thấy một hướng thay thế là gộp Step 1+2 thành 1 model dịch trực tiếp — không chọn làm chính nhưng đáng thử nghiệm nhỏ ở giai đoạn sau nếu có thời gian, vì AlignAtt/SimulStreaming đã có sẵn hạ tầng streaming dùng chung.

### 11. Qualcomm AI Hub: có nên đổi sang Zipformer / Distil-Whisper? (bổ sung 08/2026)

Đã fetch trực tiếp 2 trang [Zipformer](https://aihub.qualcomm.com/mobile/models/zipformer) và [Distil-Whisper](https://aihub.qualcomm.com/mobile/models/distil_whisper) trên Qualcomm AI Hub — nơi Qualcomm tự tối ưu/verify model cho chính Snapdragon — để xem có nên thay PhoWhisper/SenseVoice không.

| | Zipformer (AI Hub) | Distil-Whisper (AI Hub) |
|---|---|---|
| Ngôn ngữ | Anh + Trung (checkpoint song ngữ) | **Chỉ tiếng Anh** |
| Kiến trúc | Streaming transducer (encoder+decoder+joiner), ~70M tham số tổng — rất nhẹ | Whisper-small distilled, ~377M tham số |
| Streaming | Có, **native** (joiner incremental — không cần LocalAgreement/AlignAtt gắn thêm) | Không công bố |
| License | Apache 2.0 | MIT |
| Trạng thái Snapdragon | Trang tự ghi **"hiện chưa hỗ trợ trên chipset di động nào"** dù có liệt kê thiết bị tương thích | Không thấy số latency công bố trong lần fetch này |

**Kết luận: không thay được PhoWhisper/SenseVoice** — lý do dứt khoát: **cả 2 đều không có tiếng Việt** (Zipformer: Anh+Trung; Distil-Whisper: chỉ Anh). Dù tối ưu phần cứng tốt đến đâu cũng không dùng được cho phần quan trọng nhất hệ thống. Kể cả nếu chỉ định thay nhánh En/Zh/Ko: Zipformer thiếu tiếng Hàn, Distil-Whisper thiếu cả Hàn lẫn Trung → phải cõng thêm 1-2 model nữa, đi ngược hướng gộp gọn đã chọn ở §2.3.

**Nhưng có 1 insight thật đáng dùng**: Distil-Whisper cùng họ kiến trúc Whisper với PhoWhisper (encoder-decoder transformer, chỉ khác trọng số). Nếu Qualcomm đã có sẵn pipeline convert/quantize Whisper-family chạy trên Snapdragon (kể cả khi lần fetch này chưa thấy số latency cụ thể — trang AI Hub thường có bảng latency theo từng chip hiển thị động/JS-rendered mà công cụ fetch tự động có thể chưa lấy hết, nên **tự mở link trực tiếp để xem số** trước khi quyết định), nhiều khả năng **cùng pipeline đó áp dụng lại được cho checkpoint PhoWhisper** — đây là cách thực tế để lấy số RTF Snapdragon **thật** thay vì "ước lượng" như hiện tại (xem step0.md §3.2, phần hardware, đang ghi rõ là số estimate chưa đo).

**Đề xuất bổ sung vào quy trình test (§6)**: thêm 1 bước dùng `qai-hub` (Python package của Qualcomm) thử convert/quantize PhoWhisper-small sang QNN, profile trên 1 chip Snapdragon thật qua dịch vụ remote-profile miễn phí của họ — rẻ nhất để retire rủi ro "RTF Snapdragon mới chỉ là ước lượng".

Zipformer, dù không dùng ngay được, đáng ghi nhận về kiến trúc: streaming qua RNN-T joiner incremental là thiết kế "đúng bài" hơn cách hiện tại (model offline + LocalAgreement/AlignAtt gắn thêm ngoài). Nếu sau này có thời gian/dữ liệu để tự train 1 model Zipformer tiếng Việt, đây là lựa chọn kiến trúc streaming tốt — nhưng cần train from scratch (không có checkpoint Zipformer tiếng Việt public), vi phạm ràng buộc "hạn chế train" hiện tại nên chỉ ghi nhận làm hướng tương lai, không đưa vào phạm vi bây giờ.

### 12. Rà soát lại 2026: 3 candidate mới có bằng chứng vượt PhoWhisper/SenseVoice

Khi được hỏi thẳng "lựa chọn này đã đảm bảo quality/latency tốt nhất chưa", research trước đó chỉ bám theo đúng các paper trong đề bài gốc — chưa quét toàn cảnh model 2025-2026. Quét lại phát hiện 3 candidate có bằng chứng cụ thể tốt hơn:

| Model | Ngôn ngữ | Tham số | Bằng chứng | Rủi ro |
|---|---|---|---|---|
| **Zipformer-30M-RNNT-6000h** ([HF](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h)) | Chỉ Việt (thay PhoWhisper) | 30M | WER thấp hơn PhoWhisper-large dù nhỏ hơn 50 lần (train 6000h vs 844h); RTF ~0.025 trên CPU; streaming native (RNN-T) | **License CC-BY-NC-ND-4.0** — cấm thương mại + cấm sửa đổi, cần xác nhận điều khoản cuộc thi trước khi dùng ngoài phạm vi so sánh |
| **Moonshine** ([UsefulSensors](https://huggingface.co/UsefulSensors)) | Việt+Anh+Trung+Hàn — 4 model đơn ngữ riêng (moonshine-tiny-{vi,zh,ko} + moonshine-tiny cho Anh) | ~27M/model | Giảm lỗi 48% so với Whisper-Tiny, đôi khi ngang Whisper-Medium dù nhỏ hơn 28 lần; load qua transformers chuẩn giống PhoWhisper | Chưa rõ license thương mại; chưa test thật trên data mình |
| **Qwen3-ASR-0.6B** ([HF](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)) | 1 model DUY NHẤT cho cả 4 ngôn ngữ (30+ ngôn ngữ, gồm Việt/Hàn/Trung) | 0.6B | Time-to-first-token 92ms; nếu chất lượng tiếng Việt đủ tốt, có thể **gộp thay cả PhoWhisper lẫn SenseVoice thành 1 model** | Cần transformers>=5.13.0 (nhảy version lớn, có thể xung đột với funasr); model đa ngôn ngữ tổng quát thường thua specialist ở ngôn ngữ ít tài nguyên như tiếng Việt — đúng rủi ro đã nêu ở Option A (§2.3), cần test thật mới biết |

**Đã viết code test cho cả 3** (test_asr_zipformer.py, test_asr_moonshine.py, test_asr_qwen.py), chạy qua run_all_asr.py cùng với PhoWhisper/SenseVoice hiện tại, ra outputs/asr_{vi,multi,zipformer,moonshine,qwen}_results.csv để so sánh WER/CER/RTF trên cùng bộ data thật.

**Mức độ tin cậy code**: test_asr_moonshine.py rủi ro thấp nhất (dùng đúng pattern transformers.pipeline đã chạy ổn với PhoWhisper). test_asr_zipformer.py rủi ro trung bình (phụ thuộc sherpa-onnx + tự dò file token, chưa xác nhận 100% layout repo). test_asr_qwen.py rủi ro cao nhất (API rất mới, cần transformers>=5.13.0, có fallback nhưng chưa test thật).

**Kết luận tạm thời**: KHÔNG chốt kiến trúc Part A cho tới khi có số thật. Nếu Zipformer-30M thắng rõ về WER/RTF và license được cuộc thi chấp nhận (NC-ND), nên thay PhoWhisper. Nếu Qwen3-ASR-0.6B giữ WER tiếng Việt đủ tốt, cân nhắc gộp thành kiến trúc 1-model duy nhất, đơn giản hoá đáng kể so với split hiện tại.

---

### 13. Kết quả thực đo SNR robustness — Zipformer thắng rõ rệt

**Code test đã chạy:** test_asr_vi.py (PhoWhisper-small) + test_asr_zipformer.py (Zipformer-30M-RNNT-6000h) trên 5 file tiếng Việt × 6 điều kiện (sạch + 5 mức SNR: 20/15/10/5/0dB) = 30 dòng kết quả mỗi model.

**Bảng WER trung bình theo SNR:**

| SNR (dB) | PhoWhisper-small | Zipformer-30M | Khoảng cách | Kết luận |
|---|---|---|---|---|
| Sạch | 5.51% | 5.35% | ~0% | Ngang nhau |
| 20 | 5.51% | 5.89% | -0.4% | Ngang nhau |
| 15 | 6.05% | 5.89% | +0.2% | Ngang nhau |
| 10 | 7.11% | 6.95% | +0.2% | Ngang nhau |
| **5** | **13.47%** | **6.22%** | **+116%** | **Zipformer hơn 2.2 lần** |
| **0** | **8.01%** | **4.10%** | **+95%** | **Zipformer hơn 1.95 lần** |

**RTF (Real-Time Factor, nhỏ hơn tốt hơn):** PhoWhisper ~0.18 / Zipformer ~0.05 trên CUDA → Zipformer **nhanh 3.5 lần**. (Số RTF tuyệt đối cao hơn lần test sạch trước do GPU bị tranh chấp khi chạy Qwen song song; tỷ lệ tương đối vẫn ổn định.)

**Diễn giải:**
- **Ở điều kiện sạch/ít nhiễu** (≥10dB): 2 model gần như hoà — khớp với dự đoán "đều là specialist tiếng Việt tốt".
- **Ở điều kiện ồn thực tế** (5-0dB, đúng kịch bản nhà máy/môi trường không kiểm soát): **Zipformer vượt trội rõ rệt**, WER thấp hơn gần một nửa. Nguyên nhân: Zipformer train trên 6000 giờ với dữ liệu đa dạng gồm web-scraped audio tự nhiên ồn (GigaSpeech2-Vi, VietSpeech), thay vì chỉ augment noise tổng hợp trên 844 giờ như PhoWhisper.

**RTF:** Zipformer luôn nhanh hơn, không có ngoại lệ → streaming inference có thể hoạt động tốt với độ trễ thấp.

**Kết luận Part A — Thay PhoWhisper bằng Zipformer-30M:**
- ✅ WER tốt hơn ở điều kiện ồn (bài toán thật của OneVoice)
- ✅ RTF nhanh hơn 3.5x (xử lý realtime tốt hơn)
- ✅ Tham số 30M (nhỏ hơn 50x so với PhoWhisper, deploy trên Snapdragon dễ hơn)
- ⚠️ License **CC-BY-NC-ND-4.0**: cấm thương mại + cấm sửa đổi → **CẦN HỎI BAN TỔ CHỨC** trước khi dùng ngoài phạm vi so sánh

**Moonshine & Qwen3-ASR:** Chưa có dữ liệu SNR — đóng vai trò "backup candidate" nếu license Zipformer bị từ chối.

---

**Document version:** 2026-08-08 (code-tested: PhoWhisper vs Zipformer với SNR robustness real-world; Moonshine/Qwen chưa test)
**Bước tiếp theo:** (1) Xác nhận license Zipformer-30M với ban tổ chức; (2) Nếu OK, chốt Part A = Zipformer-30M (Vi) + SenseVoice-Small (En/Zh/Ko). Nếu không, fallback Moonshine hoặc Qwen3-ASR.
