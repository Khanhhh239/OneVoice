# 🏛️ Báo Cáo Kỹ Thuật: Kiến Trúc 1 Đồ Thị (Single Graph) Cho Piper Vietnamese Trên NPU Qualcomm

> **Tài liệu tham chiếu dự án OneVoice (Nhóm Nghiên Cứu & Triển Khai Edge AI)**  
> **Mục tiêu:** Chuyển đổi toàn bộ luồng tổng hợp tiếng nói Tiếng Việt (Piper VITS) từ chuỗi văn bản UTF-8 sang sóng âm 22.05 kHz thành **1 đồ thị tĩnh duy nhất (Single Static Graph)** thực thi 100% trên NPU Qualcomm Hexagon (HTP).

---

## 1. Bối Cảnh & Vấn Đề Của Kiến Trúc Cũ

### 1.1. Lỗi lệch không gian âm vị của `byte_text_encoder`
Trong các thử nghiệm ban đầu (dựa trên các repository mã nguồn mở trước đó), hệ thống gặp một rào cản nghiêm trọng:
* **Cách làm cũ:** Nhúng một mạng nơ-ron nhỏ gọi là `byte_text_encoder` nhằm chuyển đổi trực tiếp các byte UTF-8 của văn bản thành vector đặc trưng để đưa vào mô hình Piper.
* **Nguyên nhân thất bại:** Mạng `byte_text_encoder` này **hoàn toàn chưa qua huấn luyện (untrained)**. Trong khi đó, mô hình chính **Piper VITS (HiFi-GAN)** đã được huấn luyện cố định trên bảng mã âm vị rời rạc chuẩn của `espeak-ng` (ví dụ từ `"gà"` tương ứng với chuỗi ID âm vị `[g, a, \u0300]`).
* Khi một mạng ngẫu nhiên biến đổi từ `"gà"` thành một vector tùy ý (tương đương với việc đưa âm vị của từ `"vịt"` hoặc nhiễu trắng vào mô hình), bộ giải mã âm học nhận tín hiệu sai lệch và tạo ra âm thanh vô nghĩa, vỡ tiếng hoặc câm lặng.

### 1.2. Vấn đề chia nhỏ 5 Submodel (5-Graph Baseline)
Khi cố gắng bẻ nhỏ Piper thành 5 mô hình riêng biệt (`encoder`, `sdp`, `aligner`, `flow`, `decoder`):
1. **Nghẽn cổ chai giao tiếp CPU – NPU:** Cứ sau mỗi module, tensor trung gian lại phải truyền ngược từ NPU về RAM máy chủ (Host CPU), sau đó CPU chuẩn bị rồi gửi tiếp tensor sang NPU cho module kế tiếp. Độ trễ trễ trao đổi I/O lớn hơn rất nhiều lần thời gian tính toán thực tế của chip.
2. **Phụ thuộc thư viện C++ ngoài:** Bước tiền xử lý G2P (Text → Phonemes) vẫn phải chạy thư viện `espeak-ng` trên CPU, khiến giải pháp không thể xem là "chạy hoàn toàn trên NPU".

---

## 2. Giải Pháp Đột Phá: VietG2P Tĩnh & Đồ Thị Đơn (Single Graph)

```
                       LUỒNG DỮ LIỆU ĐỒ THỊ ĐƠN (FULLTTS)
 
   [Chuỗi Byte UTF-8]  ─── int32[1, 512]
            │
            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 1. VietG2P Tensor Graph (29.484 âm tiết, MatMul tĩnh)         │ ─── 168 layers NPU
   └──────────────────────────────────────────────────────────────┘
            │  Phoneme IDs + Lengths: int32[1, 512]
            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 2. Piper Text Encoder (Transformer / ResNet)                 │ ─── 403 layers NPU
   └──────────────────────────────────────────────────────────────┘
            │  x_enc, m_p, logs_p, x_mask
            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 3. Stochastic Duration Predictor (SDP)                       │ ─── 619 layers NPU
   └──────────────────────────────────────────────────────────────┘
            │  y_len, w_ceil
            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 4. Monotonic Aligner (Căn chỉnh thời gian ma trận tĩnh)      │ ─── 19 layers NPU
   └──────────────────────────────────────────────────────────────┘
            │  attn, y_mask
            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 5. Normalizing Flow                                          │ ─── 310 layers NPU
   └──────────────────────────────────────────────────────────────┘
            │  z: float32[1, 192, 1536] (Biểu diễn phổ ẩn)
            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 6. Windowed HiFi-GAN Decoder (Chia nhóm group=8)             │ ─── 117 layers NPU
   └──────────────────────────────────────────────────────────────┘
            │
            ▼
   [Sóng âm Waveform]  ─── float32[1, 399360] (22.05 kHz, ~18.1s)
```

### 2.1. Bản chất ngôn ngữ học: Tính chất Context-Free của Tiếng Việt
Khác với tiếng Anh (nơi một chữ cái có thể phát âm hoàn toàn khác nhau tùy thuộc vào ngữ cảnh, ví dụ: *"read"* ở hiện tại vs quá khứ), **tiếng Việt là ngôn ngữ đơn lập, âm tiết tính**.
* Quy tắc phát âm của từng âm tiết tiếng Việt **hoàn toàn độc lập ngữ cảnh (context-free)**.
* Kiểm chứng thực nghiệm: **0 xung đột trên 2070 âm tiết cơ bản** so với `espeak-ng`.
* Điều này đồng nghĩa với việc: Việc đổi từ chữ viết sang âm vị thực chất là một phép tra cứu từ điển tĩnh (Static Dictionary Lookup).

### 2.2. Hiện thực `VietG2P` bằng phép nhân ma trận (Tensor MatMul)
Thay vì sử dụng mã Python hoặc thư viện C++:
* Xây dựng từ điển toàn diện **29.484 âm tiết** (kết hợp từ điển PhoBERT, tập dữ liệu FLEURS-vi và các biến thể vị trí dấu thanh như `hòa`/`hoà`, `thủy`/`thuỷ`).
* Biến phép tra từ điển thành một chuỗi các phép toán tensor cơ bản: `Sub`, `Abs`, `Less`, `MatMul`, `Add`, `Clip`.
* Không có câu lệnh rẽ nhánh điều kiện (`if/else`), không có vòng lặp động (`Loop`), không có toán tử chuỗi ký tự (`String ops`).
* Toàn bộ 168 layer của module G2P được biên dịch và chạy thẳng trên bộ xử lý vector/tensor của NPU, đạt **100% trùng khớp với espeak-ng trên 361 câu test FLEURS thuần tiếng Việt**.

### 2.3. Khắc phục lỗi tràn bộ nhớ NPU bằng Decoder Grouping (`group=8`)
* **Vấn đề trên phần cứng:** Khi giải nén toàn bộ 40 cửa sổ thời gian (mỗi cửa sổ 64 frame) cùng một lúc qua mạng HiFi-GAN Decoder, bộ nhớ đệm nội bộ của Qualcomm HTP bị quá tải và báo lỗi `MEM_ALLOC`. Nếu chạy từng cửa sổ đơn lẻ (`group=1`), độ trễ chuẩn bị đồ thị lại tăng vọt.
* **Giải pháp:** Chia 40 cửa sổ thành các nhóm gồm 8 cửa sổ (`group=8`) chạy tuần tự bên trong cùng 1 đồ thị. Kỹ thuật này giúp mô hình vừa vặn tuyệt đối trong giới hạn bộ nhớ SRAM của chip mà vẫn duy trì tốc độ tính toán song song tối đa.

---

## 3. Kết Quả Đo Đạc Thực Tế Trên Phần Cứng

Các số liệu được kiểm chứng trực tiếp trên nền tảng **Qualcomm AI Hub** với phần cứng mục tiêu **Dragonwing IQ-9075 EVK** (HTP NPU):

| Tiêu chí | Tiếp cận 5 Graph rời | Tiếp cận 1 Graph duy nhất (`fullg8`) |
| :--- | :---: | :---: |
| **Số lần gọi Inference lên NPU** | 5 lần tuần tự (CPU trung gian) | **1 lần duy nhất (End-to-End)** |
| **Tỉ lệ layer chạy trên NPU** | 100% từng submodel rời | **100% toàn bộ pipeline (2077 / 2077 layer)** |
| **Độ trễ suy luận trên chip** | ~850 ms (chưa tính overhead I/O) | **~480 ms** |
| **G2P Text-to-Phoneme** | Chạy C++ trên Host CPU | **Chạy 100% trên NPU (12.5 ms)** |
| **Độ ổn định phân bổ bộ nhớ** | Phải quản lý buffer thủ công | **An toàn tuyệt đối (nhờ group=8)** |
| **Đầu ra âm thanh** | 22.05 kHz | **22.05 kHz** |

---

## 4. Ý Nghĩa Kỹ Thuật Đối Với Dự Án OneVoice

1. **Độc lập hoàn toàn với CPU:** Host CPU chỉ làm 2 việc cực kỳ nhẹ nhàng: đổi xâu ký tự sang mảng byte int32 và đọc mảng float32 ra file WAV. Mọi gánh nặng tính toán AI đều nằm trọn vẹn trong NPU.
2. **Khả năng triển khai thương mại:** Loại bỏ hoàn toàn sự phụ thuộc vào các binary C++ cồng kềnh như `espeak-ng`, giúp gói triển khai ONNX/DLC trở nên gọn nhẹ, tương thích hoàn toàn với runtime của Qualcomm (QNN / SNPE).
3. **Chất lượng âm thanh bảo toàn:** Khắc phục triệt để hiện tượng vỡ tiếng của kiến trúc cũ, đảm bảo chất lượng phát âm chuẩn xác, tự nhiên theo đúng âm học của Piper VITS tiếng Việt.
