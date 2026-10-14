# MeloTTS tiếng Trung trên NPU Dragonwing IQ-9075

Đưa MeloTTS-ZH (TTS tiếng Trung, 44.1 kHz) lên NPU Hexagon của **Dragonwing IQ-9075 EVK** qua Qualcomm AI Hub.
Kết quả: chuỗi **chữ → âm thanh chạy bằng 2 graph trên NPU** (BERT, rồi VITS), chấm bằng ASR cho **CER 4.6%** trên 8 câu.

## Kết quả (đo thật trên chip)

Chấm bằng SenseVoice fp32 nhận dạng lại audio, so với chữ đầu vào (8 câu tiếng Trung ngắn).

| Cấu hình | CER | Ghi chú |
|---|---|---|
| CPU fp32, model gốc MeloTTS | 2.6% | `results/cpu_baseline_fp32.json` |
| CPU, bản tĩnh (duration xác định, nhiễu cố định) | 4.2% | tương đương bản tĩnh dùng trên chip |
| **NPU: VITS (đặc trưng BERT từ CPU)** | **4.6%** | 2191/2191 layer NPU, **702 ms** |
| **NPU: BERT → (máy chủ mở rộng) → VITS** | **4.6%** | chuỗi 2 graph hoàn toàn trên NPU |

Hai graph trên chip:

| Graph | Layer trên NPU | Thời gian chip | Nạp model lần đầu |
|---|---|---|---|
| BERT (22 lớp, 32 token) | 690/690 | 10.7 ms | 16 s |
| VITS (encoder, duration, flow, vocoder) | 2191/2191 | 702 ms | 432 s |

Đặc trưng BERT chạy trên NPU khớp CPU với cosine ≥ 0.99993. Audio nghe thử: `results/wav_chain2_npu/` (44.1 kHz).
Job ID từng lần chạy: `docs/ket_qua_chip.md`.

## Luồng dữ liệu

```
chữ ──(máy chủ: chuẩn hoá, pinyin, id token)──► bert_ids [1,32], phones/tones [1,96]
bert_ids ──► [NPU graph 1: BERT 22 lớp] ──► h [32,1024]
h ──(máy chủ: nhân h^T với ma trận 0/1 E theo số phoneme mỗi chữ)──► bert [1,1024,96]
phones, tones, bert ──► [NPU graph 2: encoder → duration → căn chỉnh → flow → vocoder cửa sổ] ──► wav 44.1 kHz
```

Kích thước tĩnh: tối đa **96 phoneme** (khoảng 22 chữ) và **384 frame** (≈ 4.5 giây), BERT tối đa 32 token.
Câu dài hơn phải chia ở máy chủ. Chip luôn tính đủ 384 frame nên thời gian chip gần như không đổi.

## Điều chỉnh so với MeloTTS gốc

- Dùng bộ dự đoán độ dài **xác định** (`sdp_ratio=0`) thay cho `0.2`, nhiễu **cố định**. Làm CER trên CPU tăng từ 2.6% lên 4.2%.
- Độ dài và căn chỉnh viết bằng `cumsum` + so sánh (không dùng `NonZero`/`Range`).
- Vocoder chạy theo **cửa sổ 64 frame (chồng 12 frame mỗi bên)** vì chạy cả 384 frame một lần làm chip hết bộ nhớ.
  Kết quả cửa sổ giống bản đầy đủ (cosine 1.00000 trên CPU).

## Những thứ không chạy được (nói thẳng)

| Thử | Kết quả |
|---|---|
| Vocoder đủ 384 frame một lần | chip báo hết bộ nhớ (`QNN_COMMON_ERROR_MEM_ALLOC`) |
| **Gộp BERT + VITS thành một graph** | chip báo lỗi chung (`QNN_COMMON_ERROR_GENERAL`) khi chạy, nên dùng 2 graph |
| Đưa phép mở rộng `h^T @ E` vào graph VITS | chạy được nhưng **CER 18%** (tệ hơn 4.6%); phép nhân riêng lẻ trên chip lại đúng tuyệt đối, nên lỗi do cách graph được biên dịch khi gộp. Vì vậy phép này làm ở máy chủ |
| **Lượng tử hoá W8A16 và W16A16** | quantize và compile thành công, nhưng chip **không nạp được** (`MODEL_GRAPH_ERROR`) nên không chạy được |

Vì vậy bản triển khai là **fp16** (không lượng tử). Model chỉ có 52M tham số, khoảng 104 MB fp16, nên lượng tử không cần để giảm dung lượng.
Chưa tìm ra op nào gây `MODEL_GRAPH_ERROR` ở bản lượng tử; để thử tiếp có thể lượng tử riêng vocoder.

## Đo trên mẫu lớn hơn (40 mệnh đề FLEURS, CPU) — chỉnh nhiễu để tăng chất lượng

8 câu quá ít để kết luận, nên đo thêm 40 mệnh đề (`step10_variants.py`, `results/cpu_variants_40clauses.json`):

| Cấu hình | CER trung bình | Số câu có lỗi |
|---|---|---|
| Model gốc (`sdp_ratio=0.2`, nhiễu 0.6 ngẫu nhiên) | 6.8% | 18/40 |
| Bản tĩnh, nhiễu 0.6, seed 0 (**đang dùng trên chip**) | 9.1% | 21/40 |
| Bản tĩnh, nhiễu 0.6, seed 1 | 7.5% | 20/40 |
| Bản tĩnh, nhiễu 0.667, seed 0 | 9.2% | 21/40 |
| **Bản tĩnh, nhiễu 0.3, seed 0** | **6.6%** | 20/40 |

Kết luận: lỗi phát âm phân tán ở từ khó chứ không có lỗi hệ thống, và hạ nhiễu xuống 0.3 giúp bản tĩnh bằng bản gốc.
Chưa chạy lại trên chip với nhiễu 0.3 (graph trên chip hiện dùng 0.6); đổi `noise_scale` trong `melo_static.py` rồi xuất lại.

## Giới hạn

- Mới thử 8 câu ngắn, một giọng (ZH), không có đánh giá nghe chủ quan (MOS).
- Máy chủ còn làm: chuẩn hoá chữ, đổi pinyin/id, dựng ma trận E, phép mở rộng `h^T @ E` (3 triệu phép nhân), cắt audio, ghi wav.
- AI Hub chỉ chạy từng graph, không chạy mã tuỳ ý. Mỗi job thật mất khoảng 10–45 phút vì nạp model và hàng đợi.
- Số frame trên chip lệch khoảng 2–7% so với CPU do làm tròn fp16 ở bộ dự đoán độ dài (nhịp đọc hơi nhanh hơn).

## Chạy lại

Yêu cầu: Python 3.11, `pip install -r requirements.txt`, `qai-hub` đã cấu hình, và `git clone https://github.com/myshell-ai/MeloTTS` vào thư mục này (tên `MeloTTS`).
Model và BERT (`hfl/chinese-roberta-wwm-ext-large`, khoảng 1.3 GB) tự tải về lần đầu. Chạy từ thư mục `melotts_npu/` (file sinh ra ở `work/`).
Chấm điểm cần SenseVoice combined ONNX (biến `SV_ONNX`, xem nhánh `sensevoice`).

```bash
python step0_cpu_baseline.py                        # mốc CPU + CER
python step2_cpu_static_parity.py                   # bản tĩnh khớp bản gốc trên CPU
python step3_export.py 96 384 5                     # xuất VITS tĩnh (cửa sổ, nhóm 5) -> work/melo_N96_Y384_g5.onnx
python hub_melo.py g5 work/melo_N96_Y384_g5.onnx 96 384     # compile + inference + profile trên chip, CER
python step6_bert_only.py                           # xuất BERT 22 lớp (32 token) -> work/bert_L32.onnx
python hub_generic.py bert work/bert_L32.onnx work/bert_inputs.npz
python step9_chain_hostexpand.py                    # chuỗi BERT(NPU) -> máy chủ -> VITS(NPU), CER
```

## Các file

| File | Việc |
|---|---|
| `melo_static.py` | VITS tĩnh một khối + vocoder cửa sổ |
| `melo_full.py` | BERT tĩnh, ghép BERT+VITS, và `host_front` (chữ → id, ma trận E) |
| `datasets_zh.py` | Dữ liệu hiệu chuẩn lượng tử từ FLEURS tiếng Trung |
| `step0`–`step9`, `hub_*.py` | Kiểm tra CPU, xuất ONNX, chạy trên chip, chấm điểm |
| `logs/chip/` | Log chạy trên chip (đã lọc) |
| `docs/` | Nhật ký job, giải thích |
