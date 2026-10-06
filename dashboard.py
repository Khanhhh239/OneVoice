# -*- coding: utf-8 -*-
"""
Piper Vietnamese Single-Graph TTS Dashboard & Quality Benchmark
Built for testing multi-sentence synthesis, token limiting, single-load weight reuse,
and quality evaluation against reference and Qualcomm NPU hardware outputs.
"""

import os
import sys
import time
import io
import wave
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure OpenMP runtime conflict is suppressed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

HERE = os.path.dirname(os.path.abspath(__file__))
PIPER_NPU_DIR = os.path.join(HERE, "piper_npu")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tts_engine
from tts_engine import (
    PiperTTSModelManager,
    audio_to_wav_bytes,
    split_into_sentences,
    normalize_vietnamese_text,
    SAMPLE_RATE,
    DEFAULT_MAX_SYLLABLES,
)

# -------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="OneVoice Piper Vietnamese NPU Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Modern Dark/Neon Aesthetics */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00F2FE;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00F2FE;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #CBD5E1;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sentence-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background: rgba(79, 172, 254, 0.2); color: #4FACFE; border: 1px solid rgba(79, 172, 254, 0.4); }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4); }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Singleton Model Loader (cached via Streamlit)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tts_manager():
    return PiperTTSModelManager.get_instance()

with st.spinner("⏳ Đang nạp mô hình Piper Vietnamese Single Static Graph (Chỉ load 1 lần duy nhất)..."):
    mgr = load_tts_manager()


# -------------------------------------------------------------
# Plotting Helper (Waveform & Spectrogram)
# -------------------------------------------------------------
def plot_audio_waveform(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE, title: str = "Audio Waveform"):
    fig, ax = plt.subplots(figsize=(8, 1.8), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    time_axis = np.linspace(0, len(audio_np) / sample_rate, len(audio_np))
    ax.plot(time_axis, audio_np, color='#00F2FE', linewidth=0.8, alpha=0.85)
    ax.set_xlim(0, max(0.1, time_axis[-1] if len(time_axis) > 0 else 1.0))
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.set_title(title, color='#f1f5f9', fontsize=9, pad=4)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Cấu Hình Mô Hình & NPU")
    st.markdown(
        """
        <div style="padding: 10px; background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 8px; font-size: 0.82rem; margin-bottom: 15px;">
            ✅ <b>Trạng thái:</b> Đã nạp weight vào RAM.<br>
            🚀 <b>Kiến trúc:</b> Single Static Graph (2077 layers)<br>
            🎯 <b>Target Chip:</b> Qualcomm IQ-9075 / HTP NPU
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎛️ Tham Số Sinh Giọng")
    length_scale = st.slider("Tốc độ đọc (Length Scale)", min_value=0.5, max_value=2.0, value=1.0, step=0.05,
                             help="< 1.0: Đọc nhanh hơn, > 1.0: Đọc chậm hơn, 1.0: Mặc định")
    noise_scale = st.slider("Độ biến thiên ngữ điệu (Noise Scale)", min_value=0.1, max_value=1.0, value=0.667, step=0.05,
                            help="Độ phong phú biểu cảm của âm vị (SDP/Flow noise)")
    noise_w = st.slider("Độ biến thiên trường độ (Noise W)", min_value=0.1, max_value=1.0, value=0.8, step=0.05,
                        help="Độ biến thiên độ dài từng âm tiết")

    st.markdown("---")
    st.markdown("#### ✂️ Chia Câu & Giới Hạn Token")
    max_syllables = st.slider(
        "Giới hạn âm tiết mỗi câu (Max Syllables)",
        min_value=15,
        max_value=45,
        value=DEFAULT_MAX_SYLLABLES,
        step=1,
        help="Đảm bảo mỗi câu không vượt quá bộ đệm tĩnh của NPU (512 bytes / ~40 âm tiết). Tự động cắt theo dấu phẩy hoặc ngắt từ nếu câu quá dài.",
    )
    auto_normalize = st.checkbox("Chuẩn hoá số thành chữ Tiếng Việt", value=True,
                                 help="Ví dụ: '2026' -> 'hai nghìn không trăm hai mươi sáu'")
    compare_with_ref = st.checkbox("So sánh trực tiếp với Piper eSpeak ONNX gốc", value=True,
                                   help="Chạy đồng thời mô hình tham chiếu để tính Cosine Similarity và so sánh chất lượng")

    st.markdown("---")
    st.caption("OneVoice Speech Team • Piper Vietnamese NPU Evaluation Studio")


# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="main-title">🎙️ OneVoice • Piper Vietnamese NPU Studio & Benchmark</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hệ thống kiểm thử chất lượng TTS tiếng Việt: Chạy 1 Graph đồng nhất trên NPU, nạp trọng số 1 lần, tổng hợp hàng loạt câu độc lập và ghép nối liền mạch.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Multi-Sentence TTS Studio",
    "🎧 So Sánh NPU Hardware Thật",
    "🧠 Kiến Trúc 1 Graph & VietG2P",
    "📊 Benchmark Hiệu Năng & RTF",
])


# =============================================================
# TAB 1: Multi-Sentence TTS Studio
# =============================================================
with tab1:
    st.markdown("#### 📝 Nhập hoặc Dán Đoạn Văn Bản (Hỗ trợ nhiều câu / đoạn văn dài)")

    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
    with col_preset1:
        if st.button("💬 Hội thoại giao tiếp"):
            st.session_state["input_text"] = (
                "Xin chào bạn, tôi là trợ lý ảo thông minh chạy hoàn toàn trên vi xử lý NPU Qualcomm.\n"
                "Hôm nay thời tiết rất dễ chịu, bạn có kế hoạch đi dạo hay làm việc gì không?\n"
                "Chúc bạn có một ngày làm việc tràn đầy năng lượng và hiệu quả!"
            )
    with col_preset2:
        if st.button("⚡ Tin tức công nghệ AI"):
            st.session_state["input_text"] = (
                "Kiến trúc mới đã gộp toàn bộ các phân tầng từ mã hoá âm vị, căn chỉnh thời gian đến bộ giải mã sóng âm thành một đồ thị tĩnh duy nhất.\n"
                "Toàn bộ hai nghìn không trăm bảy mươi bảy tầng mạng đều được thực thi trực tiếp trên phần cứng NPU Dragonwing.\n"
                "Độ trễ trung bình trên chip chỉ đạt khoảng bốn trăm tám mươi mili giây cho mỗi câu hoàn chỉnh."
            )
    with col_preset3:
        if st.button("📚 FLEURS Test Set"):
            st.session_state["input_text"] = (
                "Công viên quốc gia có khu rừng rậm rạp, chủ yếu là cây sồi, cây vân sam và nhiều loài động vật quý hiếm.\n"
                "Các bãi biển ở đây hầu hết đều là cát trắng an toàn cho việc bơi lội và có bóng mát từ hàng dừa."
            )
    with col_preset4:
        if st.button("🔢 Câu dài có số & dấu"):
            st.session_state["input_text"] = (
                "Năm 2026, các thiết bị trí tuệ nhân tạo biên đã xử lý 100% dữ liệu giọng nói cục bộ.\n"
                "Hệ thống đạt tốc độ xử lý nhanh hơn 5 lần và giảm 60% mức tiêu thụ năng lượng so với thế hệ trước."
            )

    default_text = st.session_state.get(
        "input_text",
        "Xin chào quý vị, đây là hệ thống chuyển đổi văn bản thành giọng nói tiếng Việt.\n"
        "Mô hình đã nạp trọng số một lần duy nhất vào bộ nhớ và sẵn sàng tổng hợp nhiều câu liên tục với chất lượng cao.\n"
        "Hãy cùng trải nghiệm và đánh giá độ tự nhiên của âm thanh được tạo ra!"
    )

    input_text = st.text_area(
        "Văn bản đầu vào:",
        value=default_text,
        height=140,
        placeholder="Dán đoạn văn bản cần tổng hợp giọng nói vào đây...",
    )

    # Preview sentence split
    preview_sents = split_into_sentences(input_text, max_syllables=max_syllables)
    with st.expander(f"🔍 Xem trước phân tách ({len(preview_sents)} câu / chunks sau khi cắt theo giới hạn {max_syllables} âm tiết)"):
        for idx, s in enumerate(preview_sents):
            s_words = len(s.split())
            s_bytes = len(s.encode("utf-8"))
            st.markdown(f"**Câu {idx+1}** ({s_words} từ, {s_bytes} bytes): `{s}`")

    synth_button = st.button("⚡ Tổng Hợp Âm Thanh Hàng Loạt (1 Lần Load Weight)", type="primary", use_container_width=True)

    if synth_button and input_text.strip():
        with st.spinner("🚀 Đang thực hiện inference qua FullTTS Static Graph..."):
            t_start = time.perf_counter()
            res = mgr.batch_synthesize(
                text=input_text,
                max_syllables=max_syllables,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                compare_with_ref=compare_with_ref,
            )
            total_elapsed = time.perf_counter() - t_start

        if "error" in res:
            st.error(res["error"])
        else:
            st.success(f"🎉 Hoàn thành tổng hợp {res['num_sentences']} câu trong {total_elapsed:.2f}s!")

            # Summary Metrics Banner
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{res['num_sentences']}</div>
                    <div class="metric-label">Tổng số câu</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{res['total_audio_duration_s']}s</div>
                    <div class="metric-label">Tổng thời lượng Audio</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{res['total_inference_time_s']*1000:.0f}ms</div>
                    <div class="metric-label">Tổng thời gian xử lý</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{res['overall_rtf']:.3f}</div>
                    <div class="metric-label">Real-Time Factor (RTF)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m5:
                oov_color = "#10B981" if res['total_oov_count'] == 0 else "#F59E0B"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {oov_color}">{res['total_oov_count']}</div>
                    <div class="metric-label">Từ OOV (Ngoài bảng)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Master Combined Audio Player
            st.markdown("### 🎵 Audio Tổng Hợp Hoàn Chỉnh (Tất cả các câu nối liền)")
            master_wav_bytes = audio_to_wav_bytes(res["full_audio"], SAMPLE_RATE)
            col_audio, col_dl = st.columns([4, 1])
            with col_audio:
                st.audio(master_wav_bytes, format="audio/wav")
            with col_dl:
                st.download_button(
                    label="⬇️ Tải WAV Đầy Đủ",
                    data=master_wav_bytes,
                    file_name="onevoice_piper_full_output.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )

            # Master Waveform
            st.pyplot(plot_audio_waveform(res["full_audio"], SAMPLE_RATE, "Full Waveform (All Sentences Combined)"))

            st.markdown("---")
            st.markdown("### 📋 Chi Tiết Từng Câu & Kiểm Thử Chất Lượng")

            for idx, item in enumerate(res["results"]):
                with st.container():
                    st.markdown(f"""
                    <div class="sentence-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: 700; font-size: 1.1rem; color: #F1F5F9;">Câu #{idx+1}</span>
                            <div>
                                <span class="badge-chip badge-blue">{item['duration_s']}s Audio</span>
                                <span class="badge-chip badge-green">{item['latency_ms']}ms Infer</span>
                                <span class="badge-chip badge-amber">RTF: {item['rtf']}</span>
                            </div>
                        </div>
                        <div style="font-size: 1.05rem; color: #E2E8F0; margin-bottom: 12px;">"{item['text']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

                    wav_bytes_chunk = audio_to_wav_bytes(item["audio"], SAMPLE_RATE)

                    if compare_with_ref and "ref_audio" in item:
                        c_a1, c_a2, c_sim = st.columns([3, 3, 2])
                        with c_a1:
                            st.caption("🔹 Single Static Graph (VietG2P + Piper 100% NPU Graph)")
                            st.audio(wav_bytes_chunk, format="audio/wav")
                        with c_a2:
                            st.caption("🔸 Piper ONNX Tham Chiếu (eSpeak C++ G2P + ONNX)")
                            ref_wav_bytes = audio_to_wav_bytes(item["ref_audio"], SAMPLE_RATE)
                            st.audio(ref_wav_bytes, format="audio/wav")
                        with c_sim:
                            cos_val = item.get("cosine_sim", 0.0)
                            cos_color = "green" if cos_val > 0.9 else ("orange" if cos_val > 0.5 else "blue")
                            st.metric("Cosine Similarity", f"{cos_val:.4f}")
                            st.download_button(
                                label=f"⬇️ WAV #{idx+1}",
                                data=wav_bytes_chunk,
                                file_name=f"sentence_{idx+1}.wav",
                                mime="audio/wav",
                                key=f"dl_sent_{idx}",
                            )
                    else:
                        c_play, c_dl = st.columns([4, 1])
                        with c_play:
                            st.audio(wav_bytes_chunk, format="audio/wav")
                        with c_dl:
                            st.download_button(
                                label=f"⬇️ Tải WAV Câu {idx+1}",
                                data=wav_bytes_chunk,
                                file_name=f"sentence_{idx+1}.wav",
                                mime="audio/wav",
                                key=f"dl_sent_single_{idx}",
                            )


# =============================================================
# TAB 2: Hardware NPU Comparison (Real Silicon Recordings)
# =============================================================
with tab2:
    st.markdown("### 🎧 Thử Nghiệm Trên Phần Cứng Thật (Qualcomm Dragonwing IQ-9075 EVK)")
    st.info(
        "Dưới đây là các tệp âm thanh thu thập trực tiếp từ phần cứng NPU Qualcomm IQ-9075 EVK qua Qualcomm AI Hub. "
        "Mô hình đồ thị gộp `fullg8` thực hiện 1 lần inference duy nhất (2077/2077 layer chạy 100% trên NPU HTP)."
    )

    results_dir = os.path.join(PIPER_NPU_DIR, "results", "full_e2e_npu")
    hw_samples = [
        ("cau1", "Câu 1: 'Xin chào, tôi là trợ lý ảo...'"),
        ("cau2", "Câu 2: 'Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé!'"),
        ("cau3", "Câu 3: 'Mô hình Piper tiếng Việt chạy trên chip Qualcomm...'"),
    ]

    for prefix, desc in hw_samples:
        npu_wav = os.path.join(results_dir, f"{prefix}_NPU.wav")
        fp32_wav = os.path.join(results_dir, f"{prefix}_fp32.wav")

        st.markdown(f"#### 📌 {desc}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🚀 Đầu Ra NPU Hardware (Qualcomm IQ-9075)")
            if os.path.exists(npu_wav):
                st.audio(npu_wav, format="audio/wav")
                st.caption(f"File: `{os.path.basename(npu_wav)}` (Kích thước: {os.path.getsize(npu_wav)} bytes)")
            else:
                st.warning("Chưa tìm thấy tệp NPU wav.")

        with col2:
            st.markdown("##### 💻 Đầu Ra FP32 Tham Chiếu (CPU Reference)")
            if os.path.exists(fp32_wav):
                st.audio(fp32_wav, format="audio/wav")
                st.caption(f"File: `{os.path.basename(fp32_wav)}` (Kích thước: {os.path.getsize(fp32_wav)} bytes)")
            else:
                st.warning("Chưa tìm thấy tệp FP32 wav.")

        st.markdown("---")


# =============================================================
# TAB 3: Technical Architecture & Deep Dive
# =============================================================
with tab3:
    st.markdown("### 🧠 Phân Tích Kỹ Thuật: Từ Vấn Đề Byte-Encoder Đến Graph Gộp 1 Lần Inference")

    st.markdown("""
    #### 1. Vấn Đề Của Tiếp Cận Cũ (Untrained `byte_text_encoder`)
    * **Hiện tượng:** Bản thử nghiệm trước đây cố gắng dùng một mạng `byte_text_encoder` để chuyển trực tiếp `text_bytes -> embedding / phoneme` mà không qua huấn luyện (untrained net).
    * **Nguyên nhân cốt lõi:**
      * Mô hình âm học chính (Piper VITS) được huấn luyện trên không gian mã âm vị rời rạc chuẩn của `espeak-ng` (ví dụ từ `"gà"` được ánh xạ sang chuỗi ID âm vị cụ thể `[g, a, \u0300]`).
      * Mạng `byte_text_encoder` chưa huấn luyện gán các biểu diễn ngẫu nhiên, không khớp với bảng âm vị mà bộ giải mã VITS đã học (ví dụ tạo ra biểu diễn của `"vịt"` hoặc nhiễu ngẫu nhiên), khiến âm thanh đầu ra bị biến dạng hoặc hoàn toàn không hiểu được.
      * Tách nhỏ pipeline thành 5 submodel riêng rồi nối bằng driver CPU gây nghẽn cổ chai truyền dữ liệu giữa CPU và NPU.

    #### 2. Giải Pháp Đột Phá: `VietG2P` Bảng Tra Tĩnh Bằng Tensor MatMul
    * **Tính chất ngôn ngữ học:** Trong tiếng Việt, quy tắc phát âm của từng âm tiết độc lập theo từ viết (context-free per syllable). Đã kiểm chứng: **0 xung đột trên 2070 âm tiết**.
    * **Bảng tra ma trận tĩnh:**
      * Dựng bảng tra 29.484 âm tiết tiếng Việt + từ điển PhoBERT + biến thể thanh điệu.
      * Cài đặt G2P hoàn toàn bằng các phép toán tensor cơ bản (`Sub`, `Abs`, `Less`, `MatMul`, `Add`, `Clip`).
      * 100% layer chạy trực tiếp trên NPU HTP mà không cần gọi C++ `espeak-ng` hay mã CPU nào.
      * Độ khớp với `espeak-ng`: Đạt **100% khớp trên 361 câu test FLEURS thuần tiếng Việt**.

    #### 3. Toàn Bộ Pipeline Trong 1 Graph Duy Nhất (`FullTTS`)
    * **Dòng chảy dữ liệu end-to-end:**
      ```
      [UTF-8 Text Bytes (1, 512)]
                 ↓
           [VietG2P Tensor Graph]
                 ↓
      [Piper Text Encoder (1, 512)]
                 ↓
      [Stochastic Duration Predictor (SDP)]
                 ↓
         [Monotonic Aligner Matrix]
                 ↓
       [Normalizing Flow (1, 192, 1536)]
                 ↓
      [Windowed HiFi-GAN Decoder (Group=8)]
                 ↓
        [Waveform Audio (1, 399360)]
      ```
    * **Thông số NPU Dragonwing IQ-9075:**
      * **2077 / 2077 Layer (100%)** thực thi hoàn toàn trên NPU HTP.
      * Thời gian suy luận trên chip: **~480 ms / câu**.
      * Kỹ thuật chia nhóm Decoder (`group=8`) giải quyết triệt để lỗi phân bổ bộ nhớ (`MEM_ALLOC`) trên Qualcomm HTP.
    """)


# =============================================================
# TAB 4: Live Stress & Throughput Benchmark
# =============================================================
with tab4:
    st.markdown("### 📊 Benchmark Hiệu Năng & Tốc Độ Xử Lý Hàng Loạt")
    st.write("Đo lường thời gian thực thi, độ trễ và hệ số RTF khi tăng dần số lượng câu xử lý liên tục.")

    bench_sentences_count = st.slider("Số lượng câu chạy benchmark:", min_value=1, max_value=6, value=3)

    sample_corpus = [
        "Xin chào, đây là câu thử nghiệm số một về khả năng tổng hợp tiếng Việt.",
        "Mô hình nạp trọng số một lần duy nhất vào bộ nhớ và chạy liên tục.",
        "Kiến trúc đồ thị tĩnh giúp tối ưu hoá việc thực thi trực tiếp trên bộ xử lý đồ thị.",
        "Chất lượng âm thanh đầu ra giữ được độ tự nhiên và ngữ điệu chuẩn xác.",
        "Hệ thống tự động cắt câu và kiểm soát số lượng âm tiết để không tràn bộ đệm.",
        "Tất cả các câu được ghép nối mượt mà tạo thành một bản thu âm hoàn chỉnh.",
    ]

    if st.button("🚀 Bắt đầu Benchmark"):
        test_sents = sample_corpus[:bench_sentences_count]
        combined_text = "\n".join(test_sents)

        latencies = []
        durations = []
        rtfs = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, sent in enumerate(test_sents):
            status_text.text(f"Đang chạy câu {idx+1}/{len(test_sents)}: '{sent[:35]}...'")
            r = mgr.synthesize_sentence_full(sent, length_scale=length_scale)
            latencies.append(r["latency_ms"])
            durations.append(r["duration_s"])
            rtfs.append(r["rtf"])
            progress_bar.progress((idx + 1) / len(test_sents))

        status_text.text("✅ Hoàn thành Benchmark!")

        # Display benchmark results table
        bench_data = {
            "Câu #": [f"Câu {i+1}" for i in range(len(test_sents))],
            "Nội dung": [s[:45] + "..." for s in test_sents],
            "Thời lượng Audio (s)": [round(d, 2) for d in durations],
            "Độ trễ Latency (ms)": [round(l, 1) for l in latencies],
            "RTF": [round(rt, 4) for rt in rtfs],
        }
        st.table(bench_data)

        # Plot latency & duration
        fig, ax1 = plt.subplots(figsize=(8, 3), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#0f172a')
        x_indices = np.arange(1, len(test_sents) + 1)
        ax1.bar(x_indices - 0.2, durations, width=0.4, label="Thời lượng Audio (s)", color='#4FACFE')
        ax1.set_ylabel("Audio (s)", color='#4FACFE', fontsize=9)
        ax1.tick_params(colors='#94a3b8', labelsize=8)

        ax2 = ax1.twinx()
        ax2.plot(x_indices, [l/1000.0 for l in latencies], color='#00F2FE', marker='o', linewidth=2, label="Độ trễ Latency (s)")
        ax2.set_ylabel("Latency (s)", color='#00F2FE', fontsize=9)
        ax2.tick_params(colors='#94a3b8', labelsize=8)

        ax1.set_xticks(x_indices)
        ax1.set_xticklabels([f"Câu {i}" for i in x_indices])
        for spine in ax1.spines.values():
            spine.set_color('#334155')
        for spine in ax2.spines.values():
            spine.set_color('#334155')

        plt.title("Tương Quan Thời Lượng Audio vs Thời Gian Xử Lý", color='#f1f5f9', fontsize=10)
        st.pyplot(fig)
