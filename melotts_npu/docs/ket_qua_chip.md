# Nhật ký chạy trên chip (AI Hub, Dragonwing IQ-9075 EVK)

Job xem tại `https://workbench.aihub.qualcomm.com/jobs/<ID>/` (tài khoản AI Hub của tác giả). Log đã lọc bỏ thanh tiến trình nằm ở `logs/chip/`.
Compile fp16: `--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io`. Lượng tử hoá: thêm `--quantize_io` (không có `truncate_64bit_tensors`).

## Graph chạy được

| Thí nghiệm | Compile | Profile | Inference | Kết quả |
|---|---|---|---|---|
| VITS, vocoder cửa sổ, nhóm 5 (`g5`) | `jp3o6emzp` | `jp4e3vxlg` | `jpx0xy79p` | CER 4.6%; 2191/2191 layer NPU; 702.4 ms; nạp 432 s (`hub_g5.log`) |
| VITS, vocoder cửa sổ, nhóm 2 (`g2`) | `jgod83vd5` | `jgn1k39mp` | `jprxwe4ep` | CER 4.6%; 3511/3511 layer NPU; 703.3 ms; nạp 604 s (`hub_g2.log`) |
| BERT riêng (22 lớp, 32 token) | `jp2oel86g` | `jp0o6lq0p` | `jp8j1z9q5` (lần 2: `jp2oemxxg`) | cosine với CPU ≥ 0.99993; 690/690 layer NPU; 10.7 ms; nạp 16 s (`hub_bert.log`, `hub_bert2.log`) |
| Phép mở rộng `h^T @ E` riêng (2 cách bố trí) | `jp0o6xm2p` / `jp8j1kez5` | `jp8j1kk85` / `jp0o6xxep` | `jgj3q888p` / `jgzzn006g` | cosine 1.0 với CPU ở cả hai (`hub_expand_a.log`, `hub_expand_b.log`) |
| **Chuỗi: BERT (NPU) → máy chủ mở rộng → VITS `g5` (NPU)** | `jp3o6emzp` | — | `jpe6y0rvg` | **CER 4.6%** (`chain2.log`) |

## Graph không chạy được hoặc kém

| Thí nghiệm | Compile | Inference | Kết quả |
|---|---|---|---|
| VITS, vocoder đủ 384 frame một lần | `jp8j1zm85` | `jp4e3vvlg` | **FAILED**: `QNN_COMMON_ERROR_MEM_ALLOC` (`hub_fp16_N96_Y384.log`) |
| Gộp BERT + VITS một graph (1.37 GB) | `jpy8m6m8g` | `jgzznrmog` | **FAILED**: `QNN_COMMON_ERROR_GENERAL` sau 3117 s (`hub_full_fp16.log`) |
| VITS có phép `h^T @ E` bên trong (`vits_hE`) | `jgn1ke0vp` | `jgk68yvy5` | chạy được nhưng **CER 18.0%** (profile `jp8j172z5`: 702.5 ms, 2193 layer NPU) (`hub_vits_hE.log`, `e2e_eval.log`) |
| `g5` lượng tử W8A16 | quantize `jgj3qewxp`, compile `jgd62o2ep` | `jprxwezkp` | compile OK, **FAILED** khi nạp: `MODEL_GRAPH_ERROR` (`hub_g5_w8a16.log`) |
| `g5` lượng tử W16A16 | quantize `jpe6ykl1g`, compile `j5wy4q43g` | `jp2oel26g` | compile OK, **FAILED** khi nạp: `MODEL_GRAPH_ERROR` (`hub_g5_w16a16.log`) |

Ghi chú:
- Thời gian chip là thời gian ổn định đo bằng profile. Mỗi job thật mất khoảng 10–45 phút do nạp model và hàng đợi.
- Lượng tử hoá cần thêm bias bằng 0 cho các Conv không có bias (`preprocessPerChannel: No bias info`); đã làm trong `hub_melo.py`.
- Chuỗi hai graph lấy đầu ra BERT từ job `jp2oemxxg` (chạy trên NPU) làm đầu vào cho VITS.
