# 📘 Hướng Dẫn Sử Dụng & Kiểm Thử Dashboard OneVoice Piper NPU

Tài liệu này hướng dẫn chi tiết từng bước từ khởi động hệ thống, cấu hình tham số, thử nghiệm các kịch bản văn bản thực tế đến cách đọc và đánh giá các chỉ số âm thanh trên **OneVoice Dashboard**.

---

## 1. Khởi Động Dashboard

Mở terminal (PowerShell hoặc Command Prompt) tại thư mục dự án và chạy lệnh:

```bash
streamlit run dashboard.py
```

* **Trình duyệt tự động mở:** Truy cập địa chỉ `http://localhost:8501`.
* **Cơ chế nạp trọng số (Weight Caching):** Trong lần chạy đầu tiên, hệ thống sẽ nạp toàn bộ trọng số mô hình `FullTTS` (đồ thị tĩnh 1-Graph) và `PiperVoice` vào RAM (mất khoảng 10–15 giây). Tất cả các lần thử nghiệm tiếp theo đều **tái sử dụng bộ nhớ này**, tốc độ phản hồi tức thì và không nạp lại mô hình.

---

## 2. Giải Thích Các Tham Số Cấu Hình (Cột Bên Trái - Sidebar)

Sidebar cung cấp các công cụ tinh chỉnh giọng nói và bộ tiền xử lý:

| Tham số | Giá trị mặc định | Giải thích chi tiết & Tác động |
| :--- | :---: | :--- |
| **Tốc độ đọc**<br>(*Length Scale*) | `1.00` | • **< 1.0 (ví dụ 0.85):** Nói nhanh hơn, phù hợp đọc bản tin, thông báo.<br>• **> 1.0 (ví dụ 1.15):** Nói chậm rãi, thong thả, phù hợp kể chuyện, trợ lý giao tiếp.<br>• **= 1.0:** Tốc độ tiêu chuẩn. |
| **Độ biến thiên ngữ điệu**<br>(*Noise Scale*) | `0.667` | Điều khiển độ phong phú cao độ (Pitch) và âm sắc của bộ *Normalizing Flow*:<br>• **Kéo thấp (0.1 – 0.3):** Giọng phẳng, đều đều, phát âm rành mạch kiểu máy móc (robot).<br>• **Mặc định (0.667):** Giọng tự nhiên, có ngữ điệu lên bổng xuống trầm vừa phải.<br>• **Kéo cao (0.8 – 1.0):** Ngữ điệu nhấn nhá mạnh, biểu cảm sống động. |
| **Độ biến thiên trường độ**<br>(*Noise W*) | `0.800` | Điều khiển độ co giãn thời gian của từng âm tiết thông qua *SDP*:<br>• **Kéo thấp (0.1 – 0.3):** Độ dài các âm tiết đều chằn chặn, nhịp điệu cố định.<br>• **Kéo cao (0.8 – 0.9):** Có từ lướt nhanh, có từ ngân dài tự nhiên theo trọng âm câu. |
| **Giới hạn âm tiết mỗi câu**<br>(*Max Syllables*) | `38` | Giới hạn số từ/âm tiết tối đa cho mỗi chunk đầu vào NPU.<br>*Lý do kỹ thuật:* Đồ thị NPU có bộ đệm tĩnh cố định (512 bytes / tối đa ~40 âm tiết). Tham số này đảm bảo câu không vượt quá kích thước tensor đầu vào. |
| **Chuẩn hoá số Tiếng Việt** | `Bật (Checked)` | Tự động đọc số thành chữ chuẩn ngữ pháp tiếng Việt (ví dụ: `2026` → `hai nghìn không trăm hai mươi sáu`, `100%` → `một trăm phần trăm`), tránh bị lỗi từ ngoài bảng từ điển. |
| **So sánh với Piper eSpeak gốc** | `Bật (Checked)` | Chạy song song mô hình eSpeak ONNX CPU để tính **Cosine Similarity** và so sánh trực tiếp chất lượng âm thanh 2 bên (A/B testing). |

---

## 3. Hướng Dẫn Thử Nghiệm Từng Kịch Bản (Tab 1: Multi-Sentence Studio)

### 📌 Kịch bản 1: Thử nghiệm hội thoại giao tiếp thông thường
1. Bấm nút preset **"💬 Hội thoại giao tiếp"** phía trên ô nhập liệu.
2. Nội dung hiển thị 3 câu chào hỏi ngắn.
3. Mở mục **"🔍 Xem trước phân tách"**: Dashboard sẽ hiển thị đoạn văn được chia thành 3 câu độc lập, kèm số từ và số byte của từng câu.
4. Bấm **"⚡ Tổng Hợp Âm Thanh Hàng Loạt"**:
   * Hệ thống chỉ nạp weight 1 lần, chạy tuần tự qua 3 câu.
   * Nghe **Master Audio** trên cùng: 3 câu được ghép nối mượt mà với khoảng ngắt nghỉ tự nhiên (0.15s giữa các câu).
   * Bấm **⬇️ Tải WAV Đầy Đủ** để lưu bản ghi tổng hợp.

### 📌 Kịch bản 2: Thử nghiệm câu phức tạp có số và ký hiệu
1. Bấm nút preset **"🔢 Câu dài có số & dấu"** (chứa năm 2026, tỉ lệ 100%, 60%).
2. Giữ nguyên tuỳ chọn *Chuẩn hoá số Tiếng Việt*.
3. Bấm tổng hợp: Kiểm tra xem mô hình đọc trơn tru các cụm số mà không phát ra âm rác hay bị bỏ chữ.

### 📌 Kịch bản 3: Thử nghiệm đoạn văn dài (Copy/Paste tuỳ ý)
1. Copy một đoạn báo tin tức hoặc đoạn văn dài (khoảng 5–10 câu) dán vào ô văn bản.
2. Quan sát cơ chế tự động chia câu:
   * Nếu có câu quá dài (vượt quá 38 âm tiết), thuật toán chia câu thông minh sẽ tự động tách tại dấu phẩy hoặc ngắt cụm từ để đảm bảo mỗi câu luôn an toàn với giới hạn bộ đệm NPU.
3. Bấm tổng hợp và kiểm tra banner thống kê: Tổng số câu, Tổng thời gian audio, Hệ số RTF, Tổng số từ OOV.

---

## 4. Cách Đọc & Đánh Giá Các Chỉ Số Chất Lượng

Tại banner tổng quan và danh sách từng câu, cần chú ý các chỉ số kỹ thuật sau:

* **Thời lượng Audio (s):** Tổng độ dài sóng âm thực tế sinh ra.
* **Thời gian xử lý / Độ trễ (Latency ms):** Thời gian máy tính tính toán để tạo ra âm thanh câu đó.
* **Hệ số RTF (Real-Time Factor):**
  $$\text{RTF} = \frac{\text{Thời gian suy luận (giây)}}{\text{Thời lượng âm thanh (giây)}}$$
  * **RTF < 1.0:** Tốc độ nhanh hơn thời gian thực (sinh âm thanh nhanh hơn tốc độ người nói).
  * Trên chip NPU thật (Qualcomm IQ-9075), một câu dài ~3.5s chỉ mất ~480ms suy luận $\rightarrow$ **RTF đạt ~0.14** (nhanh gấp 7 lần thời gian thực).
* **Từ OOV (Out-of-Vocabulary):** Số lượng từ bị bỏ qua do không nằm trong bảng tra 29.484 âm tiết (thường là tên riêng tiếng nước ngoài như *Qualcomm*, *Apple*, ký hiệu lạ). Nếu OOV = 0 nghĩa là toàn bộ câu đều được ánh xạ chuẩn xác.
* **Cosine Similarity (So sánh A/B):**
  * So sánh sóng âm giữa đồ thị đơn *FullTTS* và bản tham chiếu *Piper eSpeak*.
  * Giá trị dao động gần 1.00 chứng minh 1-Graph NPU tái tạo chính xác tín hiệu âm thanh tương đương bản gốc CPU.

---

## 5. Kiểm Thử Các Tính Năng Bổ Sung

### Tab 2: 🎧 So Sánh NPU Hardware Thật
* Bấm sang Tab 2 để nghe các bản ghi trực tiếp từ phần cứng **Qualcomm Dragonwing IQ-9075 EVK** (`cau1_NPU.wav`, `cau2_NPU.wav`, `cau3_NPU.wav`).
* So sánh với file tham chiếu `_fp32.wav` cùng câu để kiểm chứng chất lượng âm học trên chip phần cứng thực tế.

### Tab 4: 📊 Benchmark Hiệu Năng & RTF
1. Chọn số lượng câu thử nghiệm (từ 1 đến 6 câu).
2. Bấm **"🚀 Bắt đầu Benchmark"**.
3. Xem bảng đo đạc chi tiết và biểu đồ so sánh tương quan giữa độ dài câu và thời gian xử lý.
