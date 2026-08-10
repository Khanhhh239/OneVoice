# OneVoice Step 2 — Machine Translation (Text → Text)

## 1. Phạm vi và trạng thái

Step 2 nhận văn bản từ ASR hoặc người dùng và dịch trực tiếp giữa bốn ngôn ngữ:

| Ngôn ngữ | Mã nội bộ | Mã NLLB |
|---|---|---|
| Tiếng Việt | `vi` | `vie_Latn` |
| Tiếng Anh | `en` | `eng_Latn` |
| Tiếng Trung giản thể | `zh` | `zho_Hans` |
| Tiếng Hàn | `ko` | `kor_Hang` |

Hệ thống hỗ trợ đủ 12 hướng có thứ tự, không cần pivot qua tiếng Anh. Phần dịch câu ngoại tuyến, benchmark, so sánh model, COMET, guard bảo toàn thông tin, INT8 và kiểm thử GPU đã hoàn thành. Streaming/AlignAtt-text đã có artifact cho 12 hướng nhưng chỉ được xem là thử nghiệm. Snapdragon/QNN chưa được xác minh trên thiết bị thật.

**Mức hoàn thiện kỹ thuật ước tính: 96%.** Phần còn thiếu để đạt mức production là human evaluation hoàn chỉnh, test domain thực tế, đo trên thiết bị đích và giải quyết giấy phép model cho mục đích thương mại.

## 2. Kiến trúc kỹ thuật

Luồng xử lý:

`source text → validate language pair → NLLB tokenizer → encoder-decoder generate → target-language forced BOS → decode → safety guard → translated text`

Các thành phần chính:

| Thành phần | File | Vai trò |
|---|---|---|
| NLLB engine | `src/step2_mt/engine.py` | Tokenize, FP16 CUDA/FP32 CPU, greedy decoding, 12 hướng |
| Quality router | `src/step2_mt/engine_router.py` | Chọn 1.3B hoặc 600M theo hướng đã benchmark |
| Benchmark | `src/step2_mt/benchmark.py` | Dịch corpus, BLEU/chrF++, latency và throughput |
| Neural/safety metrics | `src/step2_mt/score_metrics.py` | COMET và guard số, đơn vị, phủ định, màu, anchor |
| GPU profile | `src/step2_mt/profile_gpu.py` | Load time, VRAM, P50/P95/P99, batch scaling |
| INT8 runtime | `src/step2_mt/engine_ct2.py` | Chạy NLLB-600M CTranslate2 INT8 |
| Stable-prefix | `src/step2_mt/streaming_benchmark.py` | Chunk re-decode, BLEU/chrF++, AL/LAAL |
| AlignAtt-text | `src/step2_mt/alignatt_benchmark.py` | Policy dựa trên encoder-decoder cross-attention |
| Fine-tune | `src/step2_mt/finetune_nllb_lora.py` | LoRA PhoMT Vi↔En |
| Kiểm thử | `src/step2_mt/test_core.py`, `validate_mvp.py` | Regression test không cần tải model |

Thiết lập benchmark chính: greedy decoding (`num_beams=1`), batch 16, FP16 trên Tesla T4, tối đa 512 source tokens và 256 generated tokens. Với đích tiếng Trung, SacreBLEU dùng tokenizer `zh`; các đích khác dùng `13a`. Thời gian được đồng bộ CUDA trước và sau inference.

## 3. Model và quyết định triển khai

### 3.1 Model được đánh giá

| Model | Vai trò | Kết luận |
|---|---|---|
| `facebook/nllb-200-distilled-600M` | Baseline, compact và CT2 INT8 | Giữ làm fallback và profile gọn |
| `facebook/nllb-200-1.3B` | Candidate chất lượng cao | Tốt hơn ở 10/12 hướng |
| PhoMT LoRA trên 600M | Fine-tune Vi↔En, 50.000 mẫu/hướng | Reject vì không vượt benchmark khóa |

### 3.2 So sánh full FLORES-200

Mỗi model được chấm trên 1.012 câu × 12 hướng. Số dương nghĩa là 1.3B tốt hơn 600M.

| Hướng | BLEU 600M | BLEU 1.3B | ΔBLEU | chrF++ 600M | chrF++ 1.3B | ΔchrF++ |
|---|---:|---:|---:|---:|---:|---:|
| en→ko | 10,90 | 9,40 | **−1,50** | 30,71 | 30,46 | −0,25 |
| en→vi | 37,66 | 39,46 | +1,80 | 56,12 | 57,28 | +1,16 |
| en→zh | 30,28 | 24,49 | **−5,79** | 20,97 | 17,86 | −3,11 |
| ko→en | 25,11 | 26,91 | +1,80 | 51,62 | 53,28 | +1,66 |
| ko→vi | 22,71 | 24,80 | +2,09 | 43,25 | 45,33 | +2,08 |
| ko→zh | 22,26 | 22,88 | +0,62 | 16,33 | 16,71 | +0,38 |
| vi→en | 31,50 | 35,53 | **+4,03** | 55,55 | 58,95 | +3,40 |
| vi→ko | 6,73 | 7,88 | +1,15 | 23,39 | 25,66 | +2,27 |
| vi→zh | 22,78 | 23,91 | +1,13 | 16,42 | 17,18 | +0,76 |
| zh→en | 24,51 | 27,76 | **+3,25** | 51,76 | 53,93 | +2,17 |
| zh→ko | 6,37 | 7,26 | +0,89 | 23,62 | 24,31 | +0,69 |
| zh→vi | 24,38 | 26,19 | +1,81 | 45,09 | 46,40 | +1,31 |

Thay toàn bộ bằng 1.3B tăng trung bình **+0,940 BLEU**, **+1,043 chrF++** và COMET từ **0,8312 lên 0,8382**, nhưng tạo regression lớn ở `en→zh`. Vì vậy cấu hình cuối là:

- **Quality profile:** 1.3B cho 10 hướng; 600M cho `en→ko` và `en→zh`.
- **Compact/on-device research profile:** 600M/CT2 INT8 cho cả 12 hướng.
- Estimated mean BLEU gain của quality router trên FLORES: **+1,548** so với baseline 600M.

Đây là lựa chọn dựa trên FLORES, không phải cam kết rằng router sẽ tốt hơn ở mọi domain. Nếu chỉ đủ bộ nhớ cho một model, 600M là lựa chọn ổn định hơn; 1.3B toàn cục không được khuyến nghị vì regression `en→zh`.

## 4. Dataset đánh giá và huấn luyện

| Dataset | Phạm vi đã chạy | Mục đích | Ghi chú |
|---|---:|---|---|
| FLORES-200 devtest | 1.012 câu × 12 = 12.144 lượt/model | Benchmark đa miền, many-to-many, so sánh model | Reference dịch bởi người; benchmark chính |
| NTREX-128 | 1.997 câu × 12 = 23.964 lượt | Tin tức, câu dài và document context | Dữ liệu CC BY-SA 4.0 |
| MASSIVE test | 2.974 câu × 12 = 35.688 lượt | Hội thoại/trợ lý ảo và localization robustness | Không dùng BLEU để xếp hạng MT literal |
| PhoMT | 50.000 cặp/hướng cho Vi↔En | Thí nghiệm LoRA | Gated; research/education only; không phân phối lại |

Tổng benchmark baseline: **71.796 lượt dịch**. Không có câu test tự tạo trong kết quả chính. Các file chuẩn hóa nằm tại `data/mt/*.jsonl`; `data/mt/manifest.json` giữ 1.012 dòng FLORES đã căn hàng cho bốn ngôn ngữ để có thể dựng lại các cặp đánh giá.

FLORES dùng làm benchmark chính vì các ngôn ngữ được line-aligned, phù hợp many-to-many. NTREX bổ sung domain tin tức. MASSIVE là dữ liệu utterance được localize, vì vậy một bản dịch hợp nghĩa có thể khác reference về cách diễn đạt; BLEU thấp trên MASSIVE không đồng nghĩa trực tiếp với lỗi dịch.

## 5. Metrics và cách diễn giải

| Metric | Đo cái gì | Cách dùng trong Step 2 | Hạn chế |
|---|---|---|---|
| SacreBLEU | Trùng khớp word/subword n-gram | So sánh reproducible theo cùng tokenizer | Phạt paraphrase đúng; nhạy tokenization |
| chrF++ | F-score character n-gram kết hợp word n-gram | Bổ sung cho Việt/Trung/Hàn có phân đoạn khác nhau | Vẫn phụ thuộc reference |
| COMET `wmt22-comet-da` | Neural metric dùng source, hypothesis và reference | Metric semantic chính trên FLORES | Không thay thế human review; có bias từ model metric |
| Safety pass rate | Bảo toàn số, đơn vị, phủ định, màu và anchor | Regression guard cho lỗi nguy hiểm | Heuristic, không phải chứng nhận an toàn |
| P50/P95/P99 | Phân bố latency | P50 điển hình; P95/P99 tail latency | Phụ thuộc phần cứng, batch và độ dài câu |
| Throughput | Câu/giây | Đánh giá batch server | Không đại diện trải nghiệm batch 1 |
| AL/LAAL | Độ trễ đọc/ghi theo source token | So sánh policy streaming text | Không phải latency âm thanh hoặc giây trên điện thoại |
| Flicker | Mức thay đổi output tạm thời | Độ ổn định của streaming | Cần human UX test để diễn giải đầy đủ |

Guard baseline FLORES đạt safety pass trung bình **85,00%**. Candidate 1.3B tăng trung bình khoảng **0,22 điểm phần trăm**, nhưng `en→zh` giảm 5,83 điểm phần trăm; đây là thêm một lý do giữ 600M cho hướng đó. Safety pass không chứng minh bản dịch đúng hoàn toàn: nó chỉ phát hiện một số loại mất thông tin có cấu trúc.

## 6. Kết quả theo dataset

| Dataset | Mean BLEU | Mean chrF++ | Mean P50 T4 | Vai trò kết luận |
|---|---:|---:|---:|---|
| FLORES baseline 600M | 22,10 | 36,24 | 0,0452 s/câu | So sánh chất lượng chính |
| NTREX baseline 600M | 20,11 | 34,33 | 0,0504 s/câu | Robustness tin tức |
| MASSIVE baseline 600M | 13,71 | 27,81 | 0,0174 s/câu | Robustness hội thoại/localization |

Độ lệch giữa các hướng lớn. Hướng mạnh gồm `en→vi` BLEU 37,66 và `vi→en` 31,50. Hướng yếu nhất là `vi→ko` 6,73 và `zh→ko` 6,37. BLEU thấp vào tiếng Hàn cho thấy model có thể tạo câu hợp nghĩa nhưng khác reference, đồng thời cũng phản ánh chất lượng chưa đồng đều; cần human review bản ngữ trước khi dùng trong domain quan trọng.

COMET baseline theo hướng nằm trong khoảng **0,7882–0,8672**. Candidate 1.3B đạt mean **0,8382**; mức giảm mạnh nhất là `en→zh` (−0,0565), trong khi cải thiện tốt ở `vi→ko` (+0,0261), `vi→en` (+0,0232) và `ko→vi` (+0,0231).

## 7. Độ trễ, throughput và bộ nhớ

GPU profile dùng cùng 12 hướng, 5 lần lặp mỗi hướng và batch 1/4/8/16 trên Tesla T4.

| Model | Batch | P50 s/câu | P95 | P99 | Throughput câu/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| 600M | 1 | 0,4401 | 0,5566 | 0,6681 | 2,26 | 1.192 MiB |
| 600M | 8 | 0,0493 | 0,0600 | 0,0604 | 20,65 | 1.235 MiB |
| 600M | 16 | **0,0255** | 0,0302 | 0,0307 | **40,41** | 1.293 MiB |
| 1.3B | 1 | 0,8316 | 0,9806 | 0,9963 | 1,21 | 2.637 MiB |
| 1.3B | 8 | 0,0966 | 0,1118 | 0,1131 | 10,52 | 2.706 MiB |
| 1.3B | 16 | **0,0498** | 0,0567 | 0,0577 | **20,45** | 2.797 MiB |

1.3B dùng khoảng 2,2 lần model memory và chậm gần 2 lần 600M. Load time đo được 31,312 giây cho 600M và 15,033 giây cho 1.3B, nhưng không dùng hai số này để xếp hạng cold-start vì thứ tự chạy, Hugging Face cache và OS cache không được cô lập.

Quality router lazy-load và cache hai model. Nếu giữ đồng thời, riêng model weights đo được khoảng 3,8 GiB, chưa gồm framework overhead. Với request đơn lẻ, batch-1 latency mới là số gần trải nghiệm tương tác; không được dùng batch-16 P50 để mô tả latency một người dùng.

## 8. CTranslate2 INT8

NLLB-600M được chuyển sang CTranslate2 INT8, kích thước artifact **599,65 MiB**. Trên local CPU smoke benchmark 20 câu/hướng, CT2 nhanh hơn PyTorch FP32 khoảng **2,73×**, mean ΔBLEU khoảng **+0,10** và mean ΔchrF++ khoảng **+0,01**. Sai khác nhỏ có thể đến từ backend/numerical precision; đây không phải bằng chứng INT8 cải thiện chất lượng.

INT8 là profile gọn cho thử nghiệm local/edge. Chưa có ONNX→QNN và chưa đo Snapdragon, nên không được suy diễn T4/CPU thành on-device latency.

## 9. Fine-tune PhoMT LoRA

Hai adapter được train trên 50.000 mẫu/hướng và đánh giá lại trên toàn bộ 1.012 câu FLORES của hướng tương ứng:

| Hướng | BLEU trước | BLEU sau | ΔBLEU | chrF++ trước | chrF++ sau | ΔchrF++ | P50 trước | P50 sau |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vi→en | 31,50 | 29,20 | **−2,30** | 55,55 | 55,99 | +0,44 | 0,0435 | 0,0613 |
| en→vi | 37,66 | 36,42 | **−1,24** | 56,12 | 55,41 | −0,71 | 0,0484 | 0,0683 |

Adapter bị **reject** vì không vượt baseline khóa và làm tăng latency. Nguyên nhân khả dĩ gồm domain mismatch PhoMT–FLORES, tập 50.000 mẫu chưa đại diện toàn bộ corpus, hyperparameter chưa tối ưu và catastrophic specialization. Không fine-tune Vi↔Zh/Ko bằng dữ liệu tự tạo hoặc corpus chưa xác minh giấy phép.

## 10. Streaming và AlignAtt-text

Stable-prefix đã chạy đủ 12 hướng với chunk 2/4/8, 20 câu/hướng. Chunk lớn thường tăng BLEU và giảm số lần re-decode, nhưng tăng lượng source phải đọc trước khi phát output. Ví dụ `vi→en` tăng từ BLEU 35,51 ở chunk 2 lên 42,31 ở chunk 8; wall-clock P50 giảm do số lần chạy model ít hơn, không có nghĩa độ trễ đọc của người dùng giảm.

AlignAtt-text đã chạy đủ 12 hướng với `f=1/2/4`, 10 câu/hướng. Ở một số hướng, tăng `f` giảm flicker nhưng tăng AL/LAAL; ở nhiều hướng khác các cấu hình gần như giống nhau. Do sample nhỏ và đây là text-prefix simulation, kết quả chỉ chứng minh pipeline policy hoạt động, chưa chứng minh simultaneous speech production.

## 11. Điểm tốt

- Đủ 12 hướng trực tiếp, không phụ thuộc pivot.
- Benchmark lớn, reference công khai, ba loại domain và không dùng test set tự tạo làm kết quả chính.
- Kết hợp lexical, character, neural, safety và operational metrics thay vì chỉ dùng BLEU.
- Có full per-sentence artifact để audit, không chỉ giữ bảng tổng hợp.
- So sánh 600M–1.3B cùng FLORES và không deploy model lớn khi có regression.
- Fine-tune thất bại được báo cáo và reject trung thực.
- Có P50/P95/P99, throughput, VRAM, batch scaling và INT8 local profile.
- Code có unit test, runner resume và member ZIP dùng dấu `/` tương thích Kaggle.

## 12. Điểm chưa ổn và rủi ro

- Chất lượng vào tiếng Hàn thấp và variance giữa hướng lớn.
- Human review 360 câu COMET thấp nhất đã được trích nhưng chưa có reviewer Việt/Anh/Trung/Hàn điền nhãn.
- MASSIVE là localization dataset, không phải MT benchmark literal; không được dùng BLEU MASSIVE để tuyên bố SOTA.
- Safety guard mới là heuristic; chưa bao phủ hallucination, thuật ngữ chuyên ngành, sắc thái lịch sự và lỗi ngữ dụng.
- Streaming/AlignAtt-text chưa chạy với audio timestamps hoặc SimulEval speech pipeline.
- Chưa test document translation, code-switch quy mô lớn, medical/legal và input dài; model card giới hạn input huấn luyện ở 512 tokens.
- NLLB có giấy phép **CC-BY-NC 4.0**, model card định vị cho nghiên cứu và không phát hành cho production thương mại. Muốn thương mại hóa phải thay model hoặc xử lý quyền sử dụng.
- Chưa có Snapdragon/QNN measurement; profile quality giữ hai model nên không phù hợp thiết bị bộ nhớ thấp.

## 13. Hướng dẫn kiểm tra cho reviewer

Chạy từ thư mục gốc `OneVoice`:

```bash
python -m unittest src.step2_mt.test_core -v
python -m src.step2_mt.validate_mvp
```

Kết quả mong đợi: **7/7 test PASS** và `MVP validation PASS: 4 languages, 12 directions`.

Demo một câu:

```bash
python -m src.step2_mt.translate_cli --src vi --tgt ko --text "Hệ thống sẽ bắt đầu lúc 8 giờ sáng."
```

Các artifact cần đối chiếu:

- Baseline: `outputs/mt/full_kaggle/*_summary.csv` và `*_details.jsonl`.
- COMET/guard: `outputs/mt/comet/flores_metrics_*`.
- 1.3B và GPU: `outputs/mt/quality_upgrade/`.
- Quyết định router: `outputs/mt/quality_upgrade/deployment_decision.json`.
- Fine-tune: `outputs/mt/finetune_eval/compare_*.csv`.
- INT8: `outputs/mt/local_validation/` và `outputs/mt/nllb_ct2_int8/onevoice_conversion.json`.
- Human-review template: `outputs/mt/quality_upgrade/human_review_360.csv`.

Reviewer/AI không nên chỉ đọc report. Cần xác minh: số dòng detail là 12.144 FLORES, 23.964 NTREX, 35.688 MASSIVE; mỗi summary có 12 hướng; quality upgrade có 12 streaming và 12 AlignAtt CSV; ZIP integrity trả `None`.

Để kiểm tra chất lượng thủ công, lấy ngẫu nhiên câu bình thường và các câu COMET thấp trong `human_review_360.csv`, sau đó gắn nhãn: đúng nghĩa, thiếu/thừa thông tin, sai số/đơn vị, sai phủ định, sai tên riêng, không tự nhiên, sai mức lịch sự. Nên có ít nhất một người đọc được ngôn ngữ đích; AI judge chỉ là tín hiệu phụ.

## 14. Artifact chính thức

- Báo cáo này: `STEP2_MT_FINAL_REPORT.md`.
- Code chuẩn: `src/step2_mt/`.
- Dataset đã chuẩn hóa: `data/mt/`.
- Kết quả chuẩn: `outputs/mt/full_kaggle`, `comet`, `quality_upgrade`, `finetune_eval`, `local_validation`.
- Gói nộp: `artifacts/step2/OneVoice_MT_FINAL_QUALITY.zip`.

Không đóng gói PhoMT thô do điều khoản không phân phối lại. Không đóng gói model CT2 599,65 MiB và adapter LoRA bị reject vào ZIP nộp; manifest, config và kết quả đánh giá vẫn được giữ để audit.

## 15. Tài liệu tham khảo và phần sử dụng

1. NLLB Team et al., **No Language Left Behind: Scaling Human-Centered Machine Translation** — nền tảng model, FLORES-200 và đánh giá multilingual: https://arxiv.org/abs/2207.04672
2. Meta, **NLLB-200 distilled 600M model card** — intended use, giới hạn 512 tokens, domain caveat và CC-BY-NC 4.0: https://huggingface.co/facebook/nllb-200-distilled-600M
3. Goyal et al., **The FLORES-101 Evaluation Benchmark** — thiết kế benchmark human-translated, line-aligned many-to-many: https://aclanthology.org/2022.tacl-1.30/
4. Federmann et al., **NTREX-128** — test set tin tức 128 ngôn ngữ và giấy phép CC BY-SA 4.0: https://github.com/MicrosoftTranslator/NTREX
5. FitzGerald et al., **MASSIVE** — 1M utterances, 51 ngôn ngữ, 18 domain, localization bởi người dịch: https://aclanthology.org/2023.acl-long.235/
6. Post, **A Call for Clarity in Reporting BLEU Scores** — lý do dùng SacreBLEU và tokenization cố định: https://aclanthology.org/W18-6319/
7. Popović, **chrF: character n-gram F-score** — cơ sở metric chrF/chrF++: https://aclanthology.org/W15-3049/
8. Rei et al., **COMET: A Neural Framework for MT Evaluation** — neural metric tương quan với human judgement: https://aclanthology.org/2020.emnlp-main.213/
9. Doan et al., **PhoMT** — corpus Vi–En và điều khoản research/education, không phân phối lại: https://huggingface.co/datasets/vinai/PhoMT
10. CTranslate2, **Quantization documentation** — INT8 conversion, compute types và backend support: https://opennmt.net/CTranslate2/quantization.html
11. Papi et al., **Attention as a Guide for Simultaneous Speech Translation** — ý tưởng dùng attention để hướng dẫn policy đồng thời; implementation hiện tại chỉ là biến thể text-level: https://aclanthology.org/2023.acl-long.745/
12. Ansari et al., **SLTEV** — đánh giá đồng thời theo quality, latency và stability: https://aclanthology.org/2021.eacl-demos.9/
