# Step 2 — Machine Translation

NLLB-200-distilled-600M baseline cho 4 ngôn ngữ Việt/Anh/Trung/Hàn và đủ 12 chiều dịch trực tiếp.

**Artifact chốt 2026-08-10:** full benchmark baseline 71.796 lượt dịch, COMET FLORES mean 0,8312, CT2 INT8 599,65 MiB. Full FLORES bổ sung cho NLLB-1.3B cho thấy BLEU tăng ở 10/12 hướng. Profile chất lượng dùng router hybrid: 1.3B cho 10 hướng, 600M cho `en->ko` và `en->zh`; profile gọn vẫn dùng 600M. Hai adapter LoRA PhoMT **không deploy** vì BLEU giảm.

## Submission MVP

```bash
python src/step2_mt/validate_mvp.py
python src/step2_mt/translate_cli.py --device cuda --mode live
# accuracy-oriented demo (beam 4, chậm hơn):
python src/step2_mt/translate_cli.py --device cuda --mode accurate
```

CLI dịch mọi cặp trong `vi/en/zh/ko` và chạy reference-free guard cho số, đơn vị, phủ định và màu. `REVIEW` nghĩa là không nên phát câu dịch thẳng ra TTS nếu thông tin an toàn có thể đã đổi.

## Benchmark chính (không dùng dữ liệu tự tạo)

| Dataset | Quy mô | Phạm vi | Nguồn/giấy phép |
|---|---:|---|---|
| FLORES-200 devtest | 1.012 câu × 12 chiều | web/Wikipedia, bản dịch chuyên nghiệp | Meta, CC BY-SA 4.0; evaluation-only terms trên mirror mới |
| NTREX-128 | 1.997 câu × 12 chiều | 123 tài liệu tin tức, bản dịch người thật | Microsoft Translator, CC BY-SA 4.0 |
| MASSIVE 1.0 test | 2.974 câu × 12 chiều | hội thoại/trợ lý giọng nói, 18 domain, professional localization | Amazon, xem LICENSE/NOTICE của bản 1.0 |
| VietMix | ~1.000 câu Vi→En | code-switch mạng xã hội, expert-translated | gated, research-only; phải tự chấp nhận license HF |

TICO-19 có 3.071 câu domain y tế nhưng không có Việt/Hàn, nên chỉ dùng làm benchmark phụ Anh↔Trung nếu cần, không trộn vào điểm 12 chiều.

## Kaggle GPU

Tại root `OneVoice`:

```bash
pip install -r requirements.txt

# Một lệnh có resume, chạy full 3 dataset + safety + streaming 6 hướng chính:
python -m src.step2_mt.run_completion --device cuda --batch-size 16

# Nên chạy smoke test cả 3 nguồn trước (20 câu/chiều)
python src/step2_mt/run_benchmark_suite.py --smoke

# Sau khi smoke pass, chạy full lần lượt cả 3 nguồn
python src/step2_mt/run_benchmark_suite.py

# FLORES-200: tải tarball chính thức, lấy đủ devtest và dựng 12 chiều
python src/step2_mt/fetch_mt_data.py
python src/step2_mt/build_flores_all_pairs.py
python src/step2_mt/benchmark.py --dataset data/mt/flores_all_pairs.jsonl --device cuda --batch-size 16

# NTREX-128: tải trực tiếp repo chính thức Microsoft và dựng 12 chiều
python src/step2_mt/fetch_ntrex.py
python src/step2_mt/benchmark.py --dataset data/mt/ntrex_all_pairs.jsonl --device cuda --batch-size 16

# MASSIVE: hội thoại gần use case speech, chỉ dùng test split khóa
python src/step2_mt/fetch_massive.py
python src/step2_mt/benchmark.py --dataset data/mt/massive_test_all_pairs.jsonl --device cuda --batch-size 16
```

Ba full run gồm lần lượt 12.144, 23.964 và 35.688 lượt dịch. Smoke test pipeline trước khi chạy dài:

```bash
python src/step2_mt/benchmark.py --dataset data/mt/ntrex_all_pairs.jsonl --device cuda --batch-size 8 --limit-per-direction 20
```

Kết quả corpus BLEU, chrF++, mean/p50/p95 latency và hypothesis chi tiết nằm trong `outputs/mt/`.

## Metric nâng cao

Benchmark dịch và chấm neural metric là hai bước riêng để không mất hypothesis nếu COMET/XCOMET hết VRAM. Trên Kaggle nên dùng runner cô lập sau để COMET không downgrade NumPy/protobuf của kernel chính:

```bash
python -m src.step2_mt.run_comet_isolated \
  --details outputs/mt/full_kaggle/flores_details.jsonl \
  --batch-size 8 --gpus 1
```

Lần đầu cần bật Internet để cài package và tải checkpoint. Cảnh báo dependency chỉ nằm trong `.comet_env`; pipeline dịch chính không bị thay đổi. Nếu không dùng runner cô lập mới cài trực tiếp như bên dưới.

```bash
pip install -r src/step2_mt/requirements-metrics.txt

# Kiểm tra deterministic safety trước, không cần model metric/GPU
python src/step2_mt/score_metrics.py \
  --details outputs/mt/flores_all_pairs_details.jsonl \
  --skip-comet

# COMET reference-based chính thức (model khoảng 580M)
python src/step2_mt/score_metrics.py \
  --details outputs/mt/flores_all_pairs_details.jsonl \
  --comet-model Unbabel/wmt22-comet-da \
  --batch-size 16 --gpus 1

# XCOMET-XL: tùy chọn, gated, 3.5B; chấm riêng với batch nhỏ
python src/step2_mt/score_metrics.py \
  --details outputs/mt/flores_all_pairs_details.jsonl \
  --skip-comet --xcomet-model Unbabel/XCOMET-XL \
  --batch-size 2 --gpus 1
```

Trước khi dùng XCOMET-XL, đăng nhập Hugging Face, tự chấp nhận model license và chạy `huggingface-cli login` hoặc `huggingface_hub.login()` trong Kaggle. Không dùng XCOMET-XL cho commercial vì checkpoint ghi CC BY-NC-SA 4.0.

Metric xuất thêm:

- `comet_score` và COMET mean theo chiều;
- `xcomet_score`, error spans và số lỗi minor/major/critical nếu chạy XCOMET;
- exact preservation cho số, %, đơn vị, phủ định, màu và anchor dạng mã/acronym;
- `safety_pass_rate` tổng hợp và length ratio để flag output quá ngắn/dài.

Các safety metric là deterministic regression checks, không thay thế human review. Negation pass chỉ xác nhận output còn dấu hiệu phủ định, chưa chứng minh scope phủ định đúng; XCOMET và người đánh giá phải bắt lỗi đổi nghĩa tinh vi.

## Fine-tune LoRA (không trộn benchmark)

PhoMT chỉ được dùng cho nghiên cứu/giáo dục và cấm phân phối lại. Người chạy phải đọc điều khoản VinAI rồi tự chấp nhận; script không tự lách bước đồng ý.

```bash
pip install -r src/step2_mt/requirements-training.txt
python -m src.step2_mt.prepare_phomt --output-dir data/private/phomt --accept-phomt-terms

# Train riêng từng chiều; T4 x2: chạy torchrun để dùng cả hai GPU.
torchrun --nproc_per_node=2 -m src.step2_mt.finetune_nllb_lora \
  --train data/private/phomt/phomt_train.jsonl \
  --validation data/private/phomt/phomt_validation.jsonl \
  --src vi --tgt en --output outputs/mt/adapters/vi-en --epochs 1

# So trước/sau trên cùng FLORES khóa. --model có thể trỏ vào thư mục adapter PEFT.
python -m src.step2_mt.benchmark --dataset data/mt/flores_all_pairs.jsonl \
  --model outputs/mt/adapters/vi-en --src vi --tgt en \
  --device cuda --batch-size 16 --run-name flores_lora_vi_en
```

Không dùng FLORES, NTREX hoặc MASSIVE làm train. PhoMT chỉ phủ Vi↔En; Vi↔Zh và Vi↔Ko vẫn cần corpus được cấp phép/xác minh riêng, không dùng câu do dự án tự bịa để báo cải thiện.

## Streaming baseline và AlignAtt-text đo được

```bash
python -m src.step2_mt.streaming_benchmark \
  --dataset data/mt/flores_all_pairs.jsonl --src vi --tgt en \
  --chunk-tokens 2 4 8 --limit 100 --device cuda \
  --output outputs/mt/streaming_vi_en.csv
```

Đây là chunk re-decode + stable-prefix baseline và xuất BLEU/chrF++/AL/LAAL dạng token. Nó **không được gọi là AlignAtt**. AlignAtt thật cần hook cross-attention và đánh giá bằng SimulEval; chưa có số thì không ghi là hoàn thành.

NLLB text không có speech frame, vì vậy implementation bên dưới dùng đúng cross-attention nhưng đo độ trễ theo **source token**, không giả đổi sang giây:

```bash
python -m src.step2_mt.alignatt_benchmark \
  --dataset data/mt/flores_all_pairs.jsonl --src vi --tgt en \
  --f 1 2 4 --read-chunk 4 --limit 50 --device cuda \
  --output outputs/mt/alignatt_vi_en.csv
```

`f` giữ các target subword đang tập trung attention vào `f` source token cuối. Báo cáo gồm BLEU, chrF++, AL, LAAL, flicker và wall-clock P50/P95. Đây là **AlignAtt-text adaptation**, không phải con số speech-frame của SimulSeamless.

## CTranslate2 int8

```bash
python -m src.step2_mt.convert_ct2 --output outputs/mt/nllb_ct2_int8 --quantization int8
python -m src.step2_mt.benchmark --backend ct2 --model outputs/mt/nllb_ct2_int8 \
  --tokenizer facebook/nllb-200-distilled-600M --compute-type int8 \
  --dataset data/mt/flores_all_pairs.jsonl --device cpu --batch-size 8
```

Phải so BLEU/chrF++ của bản int8 với PyTorch trên đúng tập và báo dung lượng thật từ `onevoice_conversion.json`. Không suy số Snapdragon từ Kaggle/CPU.

## VietMix (hard/noisy/code-switch)

1. Đăng nhập Hugging Face và chấp nhận research-only license tại `razent/vietmix`.
2. Không đưa dataset vào repo, không redistribute và không dùng train/commercial.
3. Chỉ dùng test set để đánh giá Vi→En. Cần adapter riêng sau khi xác nhận schema tải về.

## Quy tắc đánh giá

- Không trung bình sentence BLEU; dùng corpus BLEU và corpus chrF++.
- Greedy (`--num-beams 1`) là live baseline; beam search phải là run riêng.
- GPU phải warm-up và synchronize trước/sau timing.
- Không gọi giây/câu là RTF vì input text không có duration.
- FLORES và NTREX là benchmark khóa: tuyệt đối không fine-tune trên chúng.
- Non-English→non-English của NTREX dùng translationese làm source; báo riêng và không so trực tiếp với chiều authentic English→X.
- Số Kaggle không đại diện Qualcomm; vẫn phải profile QNN/ONNX trên target hardware.

## Code legacy

`test_mt_nllb.py` và `test_mt_qwen3.py` được giữ để tái lập thí nghiệm lựa chọn model cũ 30 câu. Benchmark chính thức mới dùng `benchmark.py`.
