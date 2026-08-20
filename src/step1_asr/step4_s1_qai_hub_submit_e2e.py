"""step4_s1_qai_hub_submit_e2e.py — Submit E2E ONNX lên Qualcomm AI Hub

Biên dịch model_e2e_patched.onnx (Frontend + Encoder + Argmax) sang QNN Binary w8a16
để chạy 100% trên NPU Hexagon của Dragonwing IQ-9075 EVK.

Cách chạy:
  python src/step1_asr/step4_s1_qai_hub_submit_e2e.py
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")

TARGET_DEVICE = "Dragonwing IQ-9075 EVK"

MAX_WAV_SAMPLES = 464000   # [1, 464000]


def prepare_calib_data():
    """Tạo calibration data từ audio mẫu (fbank đã được thay bằng raw wav)."""
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    LANG_IDX = {"en": 3, "zh": 4, "ko": 7}
    TEXTNORM_ITN = 15

    samples = [r for r in manifest if r["lang"] in ["en", "zh", "ko"]][:15]

    wavs, langs, txnorms = [], [], []
    for row in samples:
        wav_path = os.path.join(ROOT, row["path"])
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        T = wav_t.shape[1]
        if T > MAX_WAV_SAMPLES:
            wav_t = wav_t[:, :MAX_WAV_SAMPLES]
        else:
            wav_t = F.pad(wav_t, (0, MAX_WAV_SAMPLES - T))
        wavs.append(wav_t.numpy())
        langs.append(np.array([LANG_IDX[row["lang"]]], dtype=np.int32))
        txnorms.append(np.array([TEXTNORM_ITN], dtype=np.int32))

    print(f"[calib] Prepared {len(wavs)} calibration samples")
    return {
        "wav": wavs,
        "language": langs,
        "textnorm": txnorms,
    }


def main():
    import qai_hub as hub

    onnx_path = os.path.join(OUT_DIR, "model_e2e_pe_fixed.onnx")
    if not os.path.exists(onnx_path):
        print(f"[ERROR] E2E ONNX not found: {onnx_path}")
        print("  Chạy trước: python src/step1_asr/step4_s1_export_e2e_onnx.py")
        return

    print(f"[submit_e2e] ONNX: {onnx_path}")

    # ── 1. Upload model ──
    print("\n[1/4] Uploading ONNX to QAI Hub ...")
    model = hub.get_model("mm6xkyz4m") # Previously uploaded
    print(f"  Model ID: {model.model_id}")

    device = hub.Device(TARGET_DEVICE)

    # ── 2. Quantize (w8a16) ──
    print("\n[2/4] Preparing calibration data ...")
    calib_raw = prepare_calib_data()
    calib_dataset = hub.upload_dataset(calib_raw)

    print("[2/4] Submitting quantize job (w8a16) ...")
    quantize_job = hub.submit_quantize_job(
        model=model,
        calibration_data=calib_dataset,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="SenseVoice_E2E_quantize_w8a16",
    )
    print(f"  Quantize job ID: {quantize_job.job_id}")
    quantize_job.wait()

    q_model = quantize_job.get_target_model()
    print(f"  Quantized model ID: {q_model.model_id}")

    # ── 3. Compile ──
    print("\n[3/4] Compiling to QNN DLC ...")
    compile_job = hub.submit_compile_job(
        model=q_model,
        device=device,
        name="SenseVoice_E2E_compile_w8a16",
        options="--target_runtime qnn_context_binary --truncate_64bit_io",
    )
    compile_job.wait()
    compile_status = compile_job.get_status().code
    if compile_status != 'SUCCESS':
        print(f"[ERROR] Compile job failed with status: {compile_status}")
        print("Please check the QAI Hub web interface for more details.")
        return
    compiled_model = compile_job.get_target_model()
    print(f"  Compiled model ID: {compiled_model.model_id}")

    # ── 4. Save job IDs ──
    job_ids = {
        "quantize_job_id": quantize_job.job_id,
        "quantize_model_id": q_model.model_id,
        "compile_job_id": compile_job.job_id,
        "compiled_model_id": compiled_model.model_id,
        "e2e_onnx": onnx_path,
    }
    out_path = os.path.join(OUT_DIR, "e2e_qai_job_ids.json")
    with open(out_path, "w") as f:
        json.dump(job_ids, f, indent=2)

    print(f"\n✅ E2E compile done! Job IDs saved to {out_path}")
    print(f"   Bước tiếp: python src/step1_asr/step4_s1_profile_e2e.py")


if __name__ == "__main__":
    main()
