# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import qai_hub as hub

ROOT = r'D:\ChuyenNganhAI\AuraTranslateEdge-OneVoice'
OUT_DIR = os.path.join(ROOT, 'outputs', 'sensevoice-e2e-onnx')
DATA_DIR = os.path.join(ROOT, 'data', 'asr')
BASE_MODEL_ID = 'mm6xkyz4m'  # model_e2e_pe_fixed.onnx
TARGET_DEVICE = 'Dragonwing IQ-9075 EVK'
MAX_WAV_SAMPLES = 464000
LANG_IDX = {'en': 3, 'zh': 4, 'ko': 7}
TEXTNORM_ITN = 15

def prepare_calib_dataset():
    print("[1/5] Chuẩn bị calibration data (15 samples)...")
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

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

    data_dict = {"language": langs, "textnorm": txnorms, "wav": wavs}
    return hub.upload_dataset(data_dict)

def prepare_inference_dataset():
    print("[1/5] Chuẩn bị inference data (3 câu test: En, Zh, Ko)...")
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_samples = []
    for lang in ["en", "zh", "ko"]:
        found = [r for r in manifest if r["lang"] == lang]
        if found:
            test_samples.append(found[0])

    wavs, langs, txnorms = [], [], []
    for row in test_samples:
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

    data_dict = {"language": langs, "textnorm": txnorms, "wav": wavs}
    dataset = hub.upload_dataset(data_dict)
    return dataset, test_samples

def ctc_decode(token_ids, sp):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().numpy()
    token_ids = np.array(token_ids).flatten()
    blank_id = 0
    collapsed = []
    prev = -1
    for t in token_ids:
        if t != blank_id and t != prev:
            collapsed.append(int(t))
        prev = t
    return sp.decode(collapsed)

def main():
    print("=" * 60)
    print("QUALCOMM AI HUB WORKBENCH — FULL SUITE SUBMISSION")
    print(f"Base Model ID: {BASE_MODEL_ID}")
    print(f"Target Device: {TARGET_DEVICE}")
    print("=" * 60)

    model = hub.get_model(BASE_MODEL_ID)
    device = hub.Device(TARGET_DEVICE)

    # ── 1. QUANTIZE JOB ──
    print("\n[Bước 1/3] SUBMITTING QUANTIZE JOB (w8a16)...")
    calib_ds = prepare_calib_dataset()
    quantize_job = hub.submit_quantize_job(
        model=model,
        calibration_data=calib_ds,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="SenseVoice_E2E_Quantize_w8a16",
    )
    print(f"  👉 QUANTIZE Job ID: {quantize_job.job_id}")
    print(f"  👉 URL: {quantize_job.url}")
    print("  Đang chờ Quantize hoàn thành...")
    quantize_job.wait()
    q_model = quantize_job.get_target_model()
    print(f"  ✅ Quantize SUCCESS! Target Model ID: {q_model.model_id}")

    # ── 2. COMPILE JOB ──
    print("\n[Bước 2/3] SUBMITTING COMPILE JOB (QNN Context Binary)...")
    compile_job = hub.submit_compile_job(
        model=q_model,
        device=device,
        name="SenseVoice_E2E_Compile_QNN",
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    print(f"  👉 COMPILE Job ID: {compile_job.job_id}")
    print(f"  👉 URL: {compile_job.url}")
    print("  Đang chờ Compile hoàn thành...")
    compile_job.wait()
    compile_status = compile_job.get_status().code
    if compile_status != 'SUCCESS':
        print(f"  ❌ Compile FAILED with status: {compile_status}")
        return
    compiled_model = compile_job.get_target_model()
    print(f"  ✅ Compile SUCCESS! Compiled Model ID: {compiled_model.model_id}")

    # ── 3. INFERENCE JOB ──
    print("\n[Bước 3/3] SUBMITTING INFERENCE JOB (Hardware Execution on Dragonwing)...")
    infer_ds, test_samples = prepare_inference_dataset()
    inference_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=infer_ds,
        name="SenseVoice_E2E_Hardware_Inference",
    )
    print(f"  👉 INFERENCE Job ID: {inference_job.job_id}")
    print(f"  👉 URL: {inference_job.url}")
    print("  Đang chờ Hardware Inference trên chip thật hoàn thành...")
    inference_job.wait()
    infer_status = inference_job.get_status().code
    print(f"  ✅ Inference Status: {infer_status}")

    # Save all new job IDs
    new_jobs = {
        "quantize_job_id": quantize_job.job_id,
        "quantize_job_url": quantize_job.url,
        "quantize_model_id": q_model.model_id,
        "compile_job_id": compile_job.job_id,
        "compile_job_url": compile_job.url,
        "compiled_model_id": compiled_model.model_id,
        "inference_job_id": inference_job.job_id,
        "inference_job_url": inference_job.url,
        "target_device": TARGET_DEVICE,
    }
    out_jobs_path = os.path.join(OUT_DIR, "e2e_qai_job_ids.json")
    with open(out_jobs_path, "w", encoding="utf-8") as f:
        json.dump(new_jobs, f, indent=2)
    print(f"\nSaved updated Job IDs to {out_jobs_path}")

    # Decode inference outputs
    try:
        import sentencepiece as spm
        sp_path = os.path.expanduser(r'~/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/chn_jpn_yue_eng_ko_spectok.bpe.model')
        if not os.path.exists(sp_path):
            sp_path = os.path.expanduser(r'~/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master/chn_jpn_yue_eng_ko_spectok.bpe.model')
        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)

        output_data = inference_job.download_output_data()
        token_ids_list = list(output_data.values())[0]

        print("\n" + "=" * 60)
        print("KẾT QUẢ INFERENCE THỰC TẾ TRÊN NPU DRAGONWING:")
        print("=" * 60)
        for i, sample in enumerate(test_samples):
            pred_tokens = token_ids_list[i]
            pred_text = ctc_decode(pred_tokens, sp)
            print(f"\n[{sample['lang'].upper()}]")
            print(f"  Reference (Gốc)  : {sample['transcript']}")
            print(f"  NPU Output (Dịch): {pred_text}")
    except Exception as e:
        print(f"\n[Decode Notice] Output downloaded, could not decode with SP: {e}")

    print("\n🎉 TOÀN BỘ SUITE ĐÃ HOÀN THÀNH VÀ HIỂN THỊ ĐỦ TRÊN WORKBENCH!")

if __name__ == '__main__':
    main()
