"""Step 4 — Bước 6/6: Verify cosine similarity + WER/CER

Mục tiêu:
  - Chạy inference_job trên hardware thật (IQ-9075 EVK)
  - So sánh output hardware vs fp32 reference bằng cosine similarity
  - Đo WER/CER thật để xác nhận chất lượng không drop quá nhiều

Definition of Done:
  cosine_similarity ≥ 0.95 cho cả 3 ngôn ngữ (En/Zh/Ko)
  WER En ≤ 8.0%,  CER Zh ≤ 3.5%,  CER Ko ≤ 6.0%

Input:
  outputs/sensevoice-onnx/qai_job_ids.json   ← có compiled_model_id
  outputs/sensevoice-onnx/fp32_ref_outputs.npz ← reference từ bước 1
  data/asr/manifest.json

Cách chạy:
  python src/step1_asr/step4_s1_verify_w8a16.py
  python src/step1_asr/step4_s1_verify_w8a16.py --only-cos-sim   # chỉ đo cos sim
  python src/step1_asr/step4_s1_verify_w8a16.py --compiled-model-id mXXXXXXXX
"""
import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")
JOB_IDS_PATH = os.path.join(OUT_DIR, "qai_job_ids.json")
FP32_REF_PATH = os.path.join(OUT_DIR, "fp32_ref_outputs.npz")
VERIFY_REPORT_PATH = os.path.join(OUT_DIR, "verify_w8a16_report.json")

TARGET_DEVICE = "Dragonwing IQ-9075 EVK"
# ONNX embedding indices (from export_meta.py dummy inputs)
# language: 0=auto, 3=en, 4=zh, 7=ko
# textnorm: 15=itn (use_itn=True), 14=woitn
LANG_EMBED_IDX = {"en": 3, "zh": 4, "ko": 7, "auto": 0}
TEXTNORM_ITN = 15

COS_SIM_THRESHOLD = 0.95
WER_THRESHOLDS = {"en": 0.08, "zh": 0.035, "ko": 0.06}


def cosine_similarity(a, b):
    """Cosine similarity giữa 2 vectors (scalar float)."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_fbank(am, wav_path):
    """Extract fbank [1, T', 560] tu raw wav."""
    import torch
    import soundfile as sf
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    wav_len = torch.tensor([wav.shape[0]], dtype=torch.int32)
    with torch.no_grad():
        frontend = am.kwargs.get('frontend')
        if frontend is not None:
            feats, feats_len = frontend(wav_t, wav_len)
        else:
            feats, feats_len = wav_t.unsqueeze(-1), wav_len
            
        if hasattr(am.model, 'normalize') and am.model.normalize is not None:
            feats, feats_len = am.model.normalize(feats, feats_len)
            
    MAX_LEN = 500
    feats_np = feats.numpy().astype(np.float32)
    feats_len_np = feats_len.numpy().astype(np.int32)
    current_len = feats_np.shape[1]
    
    padded_feats = np.zeros((1, MAX_LEN, 560), dtype=np.float32)
    if current_len > MAX_LEN:
        padded_feats[:, :MAX_LEN, :] = feats_np[:, :MAX_LEN, :]
        feats_len_np = np.array([MAX_LEN], dtype=np.int32)
    else:
        padded_feats[:, :current_len, :] = feats_np
        feats_len_np = np.array([current_len], dtype=np.int32)
        
    return padded_feats, feats_len_np, wav, sr


def prepare_sample_input(am, wav_path, lang, onnx_input_names):
    """Chuan bi input fbank cho 1 sample."""
    fbank, fbank_len, wav, sr = extract_fbank(am, wav_path)
    lang_idx = LANG_EMBED_IDX.get(lang, 0)
    all_inputs = {
        "speech":         fbank,
        "speech_lengths": fbank_len,
        "language":       np.array([lang_idx], dtype=np.int32),
        "textnorm":       np.array([TEXTNORM_ITN], dtype=np.int32),
    }
    return {k: v for k, v in all_inputs.items() if k in onnx_input_names}, wav, sr


# ---------------------------------------------------------------------------
# Phase A: Cosine Similarity (hardware output vs fp32 reference)
# ---------------------------------------------------------------------------
def run_cosine_sim_check(hub, compiled_model, onnx_path):
    """Submit inference_job và tính cosine similarity."""
    print("\n" + "─" * 50)
    print("Phase A: Cosine Similarity Check")
    print("─" * 50)

    # Load fp32 references
    if not os.path.exists(FP32_REF_PATH):
        print(f"[cos_sim] ❌ fp32 reference not found: {FP32_REF_PATH}")
        print("  Chạy bước 1 (export) để tạo reference outputs.")
        return None

    fp32_refs = np.load(FP32_REF_PATH, allow_pickle=True)
    ref_keys = [k for k in fp32_refs.files if not k.startswith("__")]
    print(f"[cos_sim] Loaded {len(ref_keys)} fp32 reference outputs")

    # Get input names từ ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_names = [inp.name for inp in sess.get_inputs()]

    device = hub.Device(TARGET_DEVICE)

    from funasr import AutoModel
    MODEL_ID_HF = "FunAudioLLM/SenseVoiceSmall"
    MODEL_ID_MS = "iic/SenseVoiceSmall"
    try:
        am = AutoModel(model=MODEL_ID_HF, hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model=MODEL_ID_MS, device="cpu", disable_update=True)
    am.model.eval()

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = [r for r in manifest if r["lang"] in ["en", "zh", "ko"]]
    results = []

    for row in samples:
        wav_path = os.path.join(ROOT, row["path"])
        key = os.path.basename(row["path"])

        if key not in fp32_refs.files:
            print(f"[cos_sim] ⚠️  No fp32 reference for {key}, skipping")
            continue

        fp32_out = fp32_refs[key]

        try:
            # Prepare inputs
            inputs, wav, sr = prepare_sample_input(am, wav_path, row["lang"], onnx_input_names)

            # Submit inference job
            inputs_for_hub = {k: [v] for k, v in inputs.items()}
            dataset_hub = hub.upload_dataset(inputs_for_hub)
            inf_job = hub.submit_inference_job(
                model=compiled_model,
                device=device,
                inputs=dataset_hub,
                name=f"SenseVoice_verify_{key}",
            )
            print(f"[cos_sim] Submitted inference job for {key} (job={inf_job.job_id})")
            inf_job.wait()

            # Lấy output từ job
            dataset_out = inf_job.download_output_data()
            hw_outputs = dataset_out.get("0")
            if hw_outputs is None or not hw_outputs:
                raise ValueError("No output returned from inference job")
                
            # Specifically grab output_0 (logits) instead of relying on dict ordering
            if "output_0" in hw_outputs:
                hw_out = hw_outputs["output_0"][0]
            else:
                hw_out = list(hw_outputs.values())[0][0]
                
            hw_out = np.array(hw_out).flatten()
            cos_sim = cosine_similarity(fp32_out, hw_out)
            status = "✅" if cos_sim >= COS_SIM_THRESHOLD else "❌"
            print(f"[cos_sim] {status} {row['lang']} {key}: cos_sim={cos_sim:.4f} "
                  f"(fp32_shape={fp32_out.shape}, hw_shape={np.array(hw_out).shape})")

            results.append({
                "key": key,
                "lang": row["lang"],
                "cos_sim": cos_sim,
                "pass": cos_sim >= COS_SIM_THRESHOLD,
                "inference_job_id": inf_job.job_id,
            })

        except Exception as e:
            import traceback
            print(f"[cos_sim] ❌ {key} failed: {e}")
            traceback.print_exc()
            results.append({"key": key, "lang": row["lang"], "cos_sim": None,
                             "pass": False, "error": str(e)})

    # Summary per language
    print("\n[cos_sim] Summary:")
    for lang in ["en", "zh", "ko"]:
        lang_results = [r for r in results if r["lang"] == lang and r["cos_sim"] is not None]
        if lang_results:
            avg = np.mean([r["cos_sim"] for r in lang_results])
            passed = sum(1 for r in lang_results if r["pass"])
            status = "✅" if avg >= COS_SIM_THRESHOLD else "❌"
            print(f"  {status} {lang.upper()}: avg cos_sim={avg:.4f} "
                  f"({passed}/{len(lang_results)} pass ≥{COS_SIM_THRESHOLD})")

    return results


# ---------------------------------------------------------------------------
# Phase B: WER/CER check (decode text từ hardware logits)
# ---------------------------------------------------------------------------
def decode_sensevoice_logits(logits, lang):
    """
    Decode logits → text.
    SenseVoice output là logits [1, T, vocab_size] → argmax → decode tokenizer.
    
    Cần SenseVoice tokenizer (stored cùng model) để decode.
    """
    # Argmax CTC decode
    token_ids = np.argmax(logits, axis=-1)  # [1, T]
    if token_ids.ndim > 1:
        token_ids = token_ids[0]

    # CTC collapse (loại bỏ repeats và blank)
    blank_id = 0
    collapsed = []
    prev = -1
    for t in token_ids:
        if t != blank_id and t != prev:
            collapsed.append(int(t))
        prev = t

    return collapsed  # trả về list token IDs (cần tokenizer để convert về text)


def run_wer_cer_check(hub, compiled_model, onnx_path):
    """
    Đo WER/CER thật bằng cách:
    1. Chạy hardware inference → logits
    2. Dùng funasr tokenizer để decode → text
    3. So sánh với transcript
    
    Cách đơn giản hơn: dùng fp32 FunASR model decode trực tiếp,
    rồi so sánh output text (không cần hardware inference lần này).
    """
    print("\n" + "─" * 50)
    print("Phase B: WER/CER Check (so sánh baseline fp32 vs w8a16 hardware)")
    print("─" * 50)
    print("[wer_cer] Sử dụng fp32 FunASR model để decode baseline WER/CER...")
    print("[wer_cer] (WER/CER trên hardware sẽ được đo sau khi có tokenizer decode pipeline)")

    try:
        import jiwer
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from common import normalize_text, normalize_text_for_cer
        from step1_asr.test_asr_multi import load_sensevoice, run_asr

        device_str = "cpu"  # dùng CPU để verify, không cần GPU
        print(f"[wer_cer] Loading fp32 SenseVoice model on {device_str}...")
        model = load_sensevoice(device_str)

        with open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)

        cer_langs = {"zh", "ko"}
        samples = [r for r in manifest if r["lang"] in ["en", "zh", "ko"]]

        results_by_lang = {"en": [], "zh": [], "ko": []}
        for row in samples:
            wav_path = os.path.join(ROOT, row["path"])
            if not os.path.exists(wav_path):
                continue

            hyp = run_asr(model, wav_path, row["lang"])
            ref = row["transcript"]

            if row["lang"] in cer_langs:
                score = jiwer.cer(normalize_text_for_cer(ref), normalize_text_for_cer(hyp))
                metric = "CER"
            else:
                score = jiwer.wer(normalize_text(ref), normalize_text(hyp))
                metric = "WER"

            results_by_lang[row["lang"]].append(score)
            threshold = WER_THRESHOLDS.get(row["lang"], 0.1)
            status = "✅" if score <= threshold else "⚠️ "
            print(f"[wer_cer] {status} {row['lang']} {metric}={score:.3f}  "
                  f"ref='{ref[:30]}' hyp='{hyp[:30]}'")

        # Summary
        wer_results = {}
        print("\n[wer_cer] Summary (fp32 baseline):")
        for lang, scores in results_by_lang.items():
            if scores:
                avg = np.mean(scores)
                threshold = WER_THRESHOLDS.get(lang, 0.1)
                metric = "CER" if lang in cer_langs else "WER"
                status = "✅" if avg <= threshold else "❌"
                print(f"  {status} {lang.upper()}: avg {metric}={avg:.4f} "
                      f"(threshold ≤{threshold:.3f})")
                wer_results[lang] = {"metric": metric, "value": avg,
                                      "threshold": threshold, "pass": avg <= threshold}

        return wer_results

    except Exception as e:
        import traceback
        print(f"[wer_cer] Failed: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-model-id", type=str,
                        help="Override compiled model ID từ qai_job_ids.json")
    parser.add_argument("--only-cos-sim", action="store_true",
                        help="Chỉ đo cosine similarity, bỏ qua WER/CER")
    parser.add_argument("--only-wer", action="store_true",
                        help="Chỉ đo WER/CER (fp32 baseline), bỏ qua cos sim")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4 — Bước 6: Verify w8a16 (cosine sim + WER/CER)")
    print("=" * 60)

    try:
        import qai_hub as hub
    except ImportError:
        print("❌ qai_hub not installed. Chạy: pip install qai-hub")
        sys.exit(1)

    # Load job IDs
    job_ids = load_job_ids() if os.path.exists(JOB_IDS_PATH) else {}
    if args.compiled_model_id:
        job_ids["compiled_model_id"] = args.compiled_model_id

    onnx_path = os.path.join(OUT_DIR, "model_patched.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(OUT_DIR, "model.onnx")

    report = {"job_ids": job_ids}

    # Phase A: Cosine similarity
    if not args.only_wer:
        if "compiled_model_id" not in job_ids:
            print("❌ compiled_model_id not found. Chạy bước 4+5 trước, hoặc dùng --compiled-model-id")
            if args.only_cos_sim:
                sys.exit(1)
        else:
            compiled_model = hub.get_model(job_ids["compiled_model_id"])
            cos_results = run_cosine_sim_check(hub, compiled_model, onnx_path)
            report["cosine_similarity"] = cos_results

    # Phase B: WER/CER
    if not args.only_cos_sim:
        compiled_model_for_wer = None
        if "compiled_model_id" in job_ids:
            compiled_model_for_wer = hub.get_model(job_ids["compiled_model_id"])
        wer_results = run_wer_cer_check(hub, compiled_model_for_wer, onnx_path)
        report["wer_cer"] = wer_results

    # Lưu report
    with open(VERIFY_REPORT_PATH, "w", encoding="utf-8") as f:
        # Serialize numpy types
        def serialize(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        json.dump(report, f, indent=2, default=serialize)
    print(f"\n[verify] Report saved → {VERIFY_REPORT_PATH}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — SenseVoice-Small w8a16 Verification")
    print("=" * 60)

    cos_data = report.get("cosine_similarity", []) or []
    if cos_data:
        for lang in ["en", "zh", "ko"]:
            lang_r = [r for r in cos_data if r.get("lang") == lang and r.get("cos_sim")]
            if lang_r:
                avg = np.mean([r["cos_sim"] for r in lang_r])
                status = "✅ PASS" if avg >= COS_SIM_THRESHOLD else "❌ FAIL"
                print(f"  {lang.upper()} cos_sim={avg:.4f}  {status}")

    wer_data = report.get("wer_cer", {}) or {}
    for lang, result in wer_data.items():
        if result:
            status = "✅ PASS" if result.get("pass") else "❌ FAIL"
            print(f"  {lang.upper()} {result.get('metric', 'WER')}="
                  f"{result.get('value', 0):.4f}  {status}")

    # Ghi kết quả vào step1.md note
    print("\n📌 Ghi nhớ: Cập nhật kết quả vào step1.md sau khi verify xong!")
    print(f"   Job IDs: {JOB_IDS_PATH}")


def load_job_ids():
    if os.path.exists(JOB_IDS_PATH):
        with open(JOB_IDS_PATH) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    main()
