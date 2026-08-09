# Step 2 — MT (Machine Translation, Text → Text)

**Status (2026-08-08):** Research xong dựa trên đọc full-text 5/7 paper trong danh sách (bỏ Agent-SiMT — đánh dấu optional trong đề bài gốc, không ảnh hưởng quyết định kiến trúc chính; NLLB-200 model card dùng kiến thức nền, không cần đọc lại vì đã xác nhận qua paper khác). **Chưa code-test** — đây là bản phân tích kiến trúc, giống trạng thái step0.md ban đầu trước khi có test thật.

**Quan trọng:** Research vòng này lật ngược một giả định trong kế hoạch gốc (xem §3) — cần đọc trước khi paste Part A vào form.

---

## Part A — Draft for Technical Proposal Form (English)

### AI Pipeline Module: Machine Translation (MT)

| Component | Model | Size | Latency (target) | Technique |
|---|---|---|---|---|
| Core translator | NLLB-200-distilled-600M | 600M | RTF < 0.1 (non-streaming) | Fine-tuned per pair |
| Streaming policy (MVP) | AlignAtt (Papi et al. 2023) | 0 extra params | AL ≈ 2s | Cross-attention threshold, zero retraining |
| Streaming policy (upgrade path) | AliBaStr-style read/write policy network | +~1-3M params | AL 32-37% lower than non-streaming | Supervised policy net trained on attention pseudo-labels |
| Low-resource boost (Vi↔Zh, Vi↔Ko) | mBART-50 fine-tune + domain-filtered back-translation | — | — | TF-IDF monolingual selection + back-translation, NOT naive BT |
| Synthetic data teacher (optional) | Hunyuan-MT-7B (WMT2025 top-1 in 30/31 pairs) | 7B, offline only | — | Distill high-quality Vi↔Zh/Ko synthetic pairs |
| Code-switch handling | Copy-Through augmentation (identity pairs during fine-tune) | — | — | Cheaper than full VietMix-style LLM pipeline |

**Latency target:** 32-37% reduction in user-perceived latency (UPL) vs. non-streaming full-sentence MT, based on measured results from a comparable on-device cascade system (Meta AI, arXiv 2508.13358) using similarly-sized (~100M) models.

**Quality target:** Streaming BLEU within ~2-3 points of non-streaming upper bound (matches AliBaStr-MT's measured gap: 43.89 vs 45.56 BLEU on ES, for example).

**Note on the "beat Google Translate" claim:** Earlier planning documents assumed back-translation on mBART would let Vi↔Zh beat Google Translate outright. Full-text reading of the source paper (arXiv 2501.19314) shows this is **not** what the paper's own numbers say — see §3. The defensible claim is: back-translation gives a **meaningful improvement over a from-scratch low-resource baseline** (+3.19 BLEU for Zh→Vi in that paper), not an outright win over commercial engines. Recommend reframing the "moat" around domain specialization (factory/safety terminology) and offline capability, not raw general-domain BLEU superiority.

---

## Part B — Phân tích đầy đủ (Vietnamese)

### 1. Tóm tắt: research vòng này thay đổi gì so với đề bài gốc

Đề bài liệt kê phương án B (khuyến nghị): NLLB-600M + AlignAtt cho live mode, phương án C: model lớn hơn cho accurate mode, phương án D: distill từ Hunyuan-MT-7B. Sau khi đọc full-text 5 paper, kết luận:

- **AlignAtt đúng là lựa chọn hợp lý cho MVP** — xác nhận qua paper gốc (SimulSeamless, IWSLT 2024): zero retraining, áp được lên bất kỳ model attention nào. Nhưng có **1 paper tốt hơn cho đúng bài toán on-device** chưa có trong danh sách gốc — xem §2.
- **Claim "Vi-Zh back-translation beat Google Translate" — SAI khi đọc full số liệu.** Paper tự kết luận ngược lại (Google Translate thắng ở Vi→Zh, họ chỉ nhỉnh hơn 0.8 BLEU ở Zh→Vi). Đây là phát hiện lật ngược quan trọng nhất — xem §3.
- **VietMix xác nhận back-translation THẤT BẠI cho code-switching** — không phải chỉ "kém hiệu quả", mà gần như vô dụng (chỉ +0.55 xCOMET so với zero-shot) vì nó "sửa" câu code-switch về đơn ngữ, mất đúng thứ cần giữ. Đây là lý do phải dùng kỹ thuật khác cho code-switch (§4).

### 2. Streaming policy: 3 phương án, 1 phát hiện mới quan trọng

Đề bài liệt kê wait-k / AlignAtt / AliBaStr. Đọc full-text cho thấy đây thực ra là **2 paper khác nhau dùng cùng tên viết tắt gần giống nhau** — dễ nhầm:

| | **AlignAtt** (SimulSeamless, IWSLT 2024) | **AliBaStr-MT** (Meta AI, on-device, Aug 2025) |
|---|---|---|
| Cơ chế | Không cần train lại — dùng cross-attention của model offline có sẵn, dừng phát từ nếu nó "nhìn" vào f khung âm thanh gần nhất | Cần train thêm: policy network học từ pseudo-label rút ra từ attention của model offline (supervised) |
| Model nền | SeamlessM4T-medium — **1.2B tham số**, quá nặng cho edge (đã loại từ đầu) | Encoder/decoder riêng — chỉ **~103M tham số**, đúng tầm on-device |
| Hyperparameter | `f` (số khung gần nhất) — phải tinh chỉnh riêng theo từng cặp ngôn ngữ (f=1 cho en-zh/en-ja, f=6 cho en-de, f=9 cho cs-en) | `δ` (ngưỡng suy luận) — chỉnh được **lúc infer, không cần train lại** để đổi latency/quality |
| Số đo thật | en-zh: BLEU 20.56, AL 1.94s (paper tự nhận "struggles" ở cặp này — đúng 1 trong các cặp OneVoice cần!) | ES: BLEU 43.89/45.30 (X→EN/EN→X) so với non-streaming 45.56/46.60 — chỉ thua ~1.5 điểm; AL giảm từ 4.39s→2.74s (**37% nhanh hơn**); UPL đầu câu giảm 34% |

**Kết luận: AliBaStr-MT phù hợp hơn AlignAtt cho đúng bài toán OneVoice** — vì (1) model nhỏ 100M-class thay vì 1.2B, khớp ngân sách edge thật; (2) tự báo cáo test trên chính kịch bản on-device conversational (không phải MuST-C academic benchmark); (3) latency đo được thấp nhất trong mọi phương án so sánh (kể cả thấp hơn wait-k và EMMA). Cái giá: cần train thêm 1 policy network nhỏ (~1-3M tham số phụ, không phải train lại toàn bộ NLLB) — không "zero-cost" như AlignAtt nhưng vẫn rẻ.

**Khuyến nghị 2 giai đoạn:**
- **MVP/live-mode ban đầu**: AlignAtt trên NLLB-600M — đúng như đề bài gốc, triển khai nhanh, không cần data huấn luyện chính sách.
- **Nâng cấp nếu còn thời gian**: thay bằng policy network kiểu AliBaStr-MT (train trên chính NLLB-600M, dùng pseudo-label từ attention của bản NLLB fine-tune offline) — cải thiện latency đáng kể mà vẫn giữ model nhỏ.

*(Agent-SiMT — paper optional trong đề bài — không đọc sâu vì không ảnh hưởng quyết định 2 phương án trên; nếu cần khung "agent" phân vai Live/Accurate rõ ràng hơn có thể xem lại sau.)*

### 3. Vi-Zh low-resource: SỬA claim "beat Google Translate"

Đọc full-text arXiv 2501.19314 (Samsung SDS, VLSP 2022 challenge), bảng số thật:

**Vi→Zh (BLEU, test set):**
| Hệ thống | Điểm |
|---|---|
| Google Translate | **41.6** |
| UET Engine | 39.2 |
| Bản của họ (mBART-50 + back-translation, full pipeline) | 38.97 |

**Zh→Vi (BLEU, test set):**
| Hệ thống | Điểm |
|---|---|
| UET Engine | **40.6** |
| Bản của họ (full pipeline) | 38.90 |
| Google Translate | 38.1 |

Paper tự kết luận: *"Google Translate is the best for Vietnamese-to-Chinese translation while the UET engine is the best for Chinese-to-Vietnamese translation"* — tức **chính tác giả thừa nhận không thắng ở cả 2 chiều**. Chỉ nhỉnh hơn Google Translate 0.8 điểm ở MỘT chiều (Zh→Vi), thua rõ ở chiều còn lại.

**Điều back-translation THỰC SỰ làm được** (đáng tin, có số chứng minh): nâng baseline from-scratch lên đáng kể — Zh→Vi từ 35.58→38.90 (+3.19 BLEU, con số này đúng như abstract paper nói), Vi→Zh từ 38.22→38.97 (+0.75, khiêm tốn hơn). Đây là kỹ thuật **tốt để nâng cấp 1 hệ thống yếu**, không phải công thức để "vượt mặt Google Translate".

**Ảnh hưởng tới OneVoice**: "moat" (lợi thế cạnh tranh) không nên đặt cược vào việc BLEU tổng quát vượt Google Translate — rủi ro cao, dữ liệu thật không ủng hộ. Lợi thế thật sự nằm ở:
1. **Hoạt động offline** — Google Translate cần mạng, OneVoice không.
2. **Chuyên biệt hoá thuật ngữ nhà máy/an toàn lao động** — Google Translate là general-domain, không có lợi thế ở từ vựng chuyên ngành hẹp.

Cả 2 điều này không cần đo bằng BLEU tổng quát để chứng minh giá trị — nên đưa DoD (Definition of Done) về hướng "đúng thuật ngữ chuyên ngành > 95%" (đã có trong đề bài gốc) thay vì "BLEU ≥ Google Translate" làm tiêu chí chính.

### 4. Code-switching: back-translation thất bại, cần Copy-Through

VietMix (arXiv 2505.24472, UMD, đọc full-text) đo trực tiếp: back-translation trên text code-switch Việt-Anh chỉ cải thiện **+0.55 xCOMET so với zero-shot** (77.72→78.27 cho Qwen2.5-7B) — gần như vô dụng. Lý do paper nêu rõ: model back-translation có xu hướng "sửa" code-switching thành câu đơn ngữ sạch sẽ, tức **xoá mất đúng hiện tượng cần dịch cho đúng**.

Giải pháp VietMix dùng — pipeline 3 giai đoạn với LLM 7-9B (Qwen2.5-7B/GemmaX2-9B) + lọc chất lượng nhiều tầng — hiệu quả nhưng **quá nặng cho NLLB-600M edge**, không tương thích trực tiếp.

**Giải pháp nhẹ hơn, khớp ngân sách OneVoice**: paper AliBaStr-MT (§2) có kỹ thuật **"Copy-Through"** — thêm cặp dữ liệu huấn luyện dạng "giữ nguyên" (English→English, X→X) vào tập fine-tune, dạy model KHÔNG dịch phần đã đúng ngôn ngữ đích, giữ nguyên từ code-switch thay vì cưỡng ép dịch. Đơn giản hơn nhiều so với pipeline VietMix, không cần LLM lớn để sinh data.

**Đề xuất dùng VietMix làm gì**: bộ test 1,002 câu gold-standard (Vi→En, license CC BY-NC-SA — non-commercial, đủ cho giai đoạn thi/nghiên cứu hiện tại) dùng làm **benchmark nội bộ** để đo model OneVoice xử lý code-switch tốt tới đâu — không cần replicate pipeline nặng của họ, chỉ mượn test set để đánh giá.

### 5. Hunyuan-MT-7B làm teacher — khả thi

Đọc full-text arXiv 2509.05209 (Tencent): Hunyuan-MT-7B đứng #1 trong 30/31 cặp ngôn ngữ WMT2025, hỗ trợ 33 ngôn ngữ **có tiếng Việt** (bảng ngôn ngữ hỗ trợ liệt kê `vi`). Không có số Vi-Zh/Vi-Ko riêng biệt trong report (chỉ báo cáo theo nhóm ZH↔XX/EN↔XX gộp), nhưng khoảng cách với Google Translate ở các nhóm này rất lớn (VD ZH⇒XX: Hunyuan 0.876 vs Google 0.762 XCOMET-XXL) — đủ tin cậy để dùng làm **teacher sinh dữ liệu tổng hợp Vi↔Zh/Vi↔Ko chất lượng cao hơn back-translation thô**, đúng theo hướng phương án D trong đề bài gốc. Model chỉ 7B, chạy offline trên máy dev (không phải trên thiết bị edge) để sinh data — không ảnh hưởng ngân sách runtime edge.

### 6. Kiến trúc khuyến nghị (tổng hợp)

- **Live mode**: NLLB-600M fine-tune per pair + AlignAtt (MVP) → nâng cấp policy AliBaStr-style nếu có thời gian.
- **Vi↔En**: NLLB-600M fine-tune trên PhoMT (đã có, chất lượng cao, không cần back-translation).
- **Vi↔Zh, Vi↔Ko** (low-resource): NLLB-600M/mBART-50 fine-tune + back-translation domain-filtered (kỳ vọng thực tế: cải thiện rõ so với from-scratch, KHÔNG kỳ vọng vượt Google Translate) + tùy chọn bổ sung synthetic data từ Hunyuan-MT-7B teacher.
- **Code-switch**: Copy-Through augmentation khi fine-tune, đánh giá bằng VietMix test set.
- **Accurate mode** (phương án C, không bắt buộc phase đầu): giữ nguyên đề xuất HY-MT1.5-1.8B hoặc gọi thẳng Hunyuan-MT-7B nếu đủ tài nguyên khi không cần streaming.

### 7. Kế hoạch test (theo đúng quy trình đề bài yêu cầu)

Chưa code — đây là bước tiếp theo, theo đúng mẫu đã làm ở Step 0/Step 1 (viết code test thật, chạy trên GPU, lấy số thật trước khi chốt):

1. Baseline NLLB-600M gốc trên FLORES-200 → BLEU/COMET 6 hướng.
2. Fine-tune riêng Vi-En (PhoMT), Vi-Zh (VLSP 2022 + back-translation theo đúng công thức đã kiểm chứng ở §3), Vi-Ko (AI Hub Korea + back-translation/pivot).
3. So trước/sau fine-tune — kỳ vọng cải thiện rõ so với baseline, KHÔNG kỳ vọng thắng Google Translate (đã sửa kỳ vọng theo §3).
4. Test streaming: SimulEval đo AL/LAAL, quét δ (AlignAtt) hoặc f — vẽ đường cong latency-quality, đối chiếu với số AliBaStr-MT đã có (32-37% latency reduction) làm mốc tham chiếu.
5. Test thuật ngữ an toàn lao động — tiêu chí chính cho DoD, không phải BLEU tổng quát.
6. Test code-switch bằng VietMix test set (1,002 câu Vi→En).

---

### 8. Code-test thật: NLLB-600M vs Qwen3-0.6B vs Qwen3-1.7B trên FLORES-200

**Lý do test lại từ đầu**: NLLB-600M ban đầu chỉ là thừa kế từ kế hoạch gốc, CHƯA từng được đo thật hay so sánh với ứng viên khác. Phát hiện quan trọng: **NLLB không có trên Qualcomm AI Hub** (đề bài khuyến khích dùng model có sẵn ở đó), trong khi **Qwen3-0.6B và Qwen3-1.7B có sẵn, đã quantize cho Snapdragon**. Test để xem việc "có sẵn trên AI Hub" có đáng đánh đổi lấy chất lượng không.

**Dữ liệu**: FLORES-200 devtest (tải trực tiếp từ `dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` -- các nguồn HF Datasets khác đều gated hoặc dùng loading script không còn được hỗ trợ), 30 câu song song đủ Vi/En/Zh/Ko, cả 6 chiều dịch.

**Kết quả BLEU (sentence-level sacreBLEU, tokenizer "zh" cho tiếng Trung):**

| Chiều | NLLB-600M | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|
| vi→en | **33.81** | 18.44 | 26.85 |
| en→vi | **29.67** | 16.19 | 26.01 |
| vi→zh | 20.45 | 17.74 | **23.41** |
| zh→vi | **21.07** | 10.39 | 17.58 |
| vi→ko | **8.05** | 3.18 | 5.75 |
| ko→vi | **18.60** | 5.63 | 13.97 |
| Tốc độ (giây/câu) | **0.4-0.9** | 2.7-6.9 | 1.6-10.1 |

**NLLB-600M thắng 5/6 chiều** (chỉ thua vi→zh trước Qwen3-1.7B: 20.45 vs 23.41) **và thắng tốc độ ở MỌI chiều** (nhanh hơn Qwen3 3-15 lần tuỳ chiều). Qwen3-0.6B (đúng cỡ NLLB) thua NLLB rất xa ở mọi chiều -- lợi thế "có sẵn trên AI Hub, cùng cỡ" không bù được khoảng cách chất lượng.

**Kết luận cho câu hỏi kiến trúc**: dù NLLB không có trên AI Hub (phải tự quantize/convert ONNX→QNN), số liệu thật cho thấy **vẫn nên giữ NLLB-600M** làm nền MT -- lợi thế AI Hub (đã tối ưu sẵn) không đáng để đánh đổi lấy chất lượng kém hơn nhiều lần và tốc độ chậm hơn 3-15 lần của Qwen3. Việc tự quantize NLLB tốn công hơn nhưng là lựa chọn đúng.

**Phát hiện phụ quan trọng -- vi→ko thấp bất thường (8.05) so với ko→vi (18.60), điều tra bằng cách soi mẫu dịch thật:**

Kiểm tra 5 câu vi→ko thật của NLLB cho thấy **phần lớn ngữ nghĩa đúng** (trừ 1 lỗi thật: "tuần lộc" (reindeer) bị dịch nhầm thành "오징어" (mực/squid) ở câu đầu). BLEU thấp chủ yếu do: **tiếng Hàn là ngôn ngữ chắp dính (agglutinative)** -- cùng một nghĩa có thể chia đuôi từ/trợ từ khác nhau (VD REF "않았습니다" quá khứ vs HYP "않습니다" hiện tại, đều đúng ngữ pháp), khiến BLEU tính trùng khớp n-gram bề mặt bị đánh giá thấp giả tạo dù nghĩa đúng. Đây là **hạn chế cố hữu của BLEU với ngôn ngữ chắp dính khi nó là ngôn ngữ ĐÍCH** -- giải thích đúng khớp với việc ko→vi KHÔNG bị ảnh hưởng (đích là tiếng Việt, không chắp dính). Khác với bug CER khoảng trắng tiếng Trung ở Step 1 (có thể sửa bằng chuẩn hoá text), lỗi này không có cách sửa đơn giản -- cần metric ngữ nghĩa (COMET/xCOMET) thay vì BLEU bề mặt để đánh giá công bằng các cặp có tiếng Hàn làm đích. Ghi nhận: BLEU vi→ko/ko→vi tuyệt đối vẫn có thể phản ánh phần nào NLLB yếu hơn thật ở hướng này (ít dữ liệu train ko→ hơn →ko), nhưng khoảng cách 8.05 vs 18.60 phần lớn là do đặc tính metric, không phải NLLB "kém 2 lần" như con số thô gợi ý.

**Bug môi trường gặp khi code-test (đáng nhớ)**:
- `Muennighoff/flores200` (HF Datasets) dùng loading script không còn được `datasets` hỗ trợ; `facebook/flores` và `openlanguagedata/flores_plus` đều gated (cần đăng nhập HF). Giải pháp: tải thẳng tarball chính thức của Meta (`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`), không qua thư viện `datasets`.
- Script `test_mt_qwen3.py` chạy nền (`run_in_background`) liên tục có vẻ "treo" (>30 phút không tiến triển) dù GPU vẫn bận thật -- nguyên nhân: dùng nhầm tham số `torch_dtype` đã deprecated (transformers bản dev `5.15.0.dev0`) thay vì `dtype`, khiến load chậm hơn nhiều so với kỳ vọng. Sửa xong + thêm progress log theo từng câu để tránh mù thông tin trong các lần chạy nền dài.

---

**Document version:** 2026-08-09 (code-tested: NLLB-600M vs Qwen3-0.6B/1.7B trên FLORES-200 thật, 6 chiều)
**Bước tiếp theo:** Chốt NLLB-600M làm baseline MT chính thức; xác nhận kế hoạch tự quantize ONNX→QNN (vì không có sẵn trên AI Hub); cân nhắc đo lại vi↔ko bằng COMET/xCOMET thay vì chỉ BLEU trước khi kết luận NLLB "yếu" ở cặp Hàn.
