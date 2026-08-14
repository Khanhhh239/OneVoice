"""Step 4 -- Buoc 1/6: Export SenseVoice-Small -> ONNX (fp32)

Su dung chinh xac export_rebuild_model tu funasr/models/sense_voice/export_meta.py
(API chinh thuc cua FunASR team, khong phai tu viet).

KEY INSIGHT tu export_meta.py:
  - Input ONNX KHONG phai raw waveform -- la fbank features [B, T', 560]
    (frontend da duoc goi truoc o ngoai, ONNX model nhan fbank truc tiep)
  - 4 inputs rieng: speech [B,T',560], speech_lengths [B], language [B], textnorm [B]
  - language: embedding index (0=auto, 3=en, ...) -- KHAC voi vocab ID
  - textnorm: 15=itn (use_itn=True), 14=woitn

Verified tu dummy inputs:
  speech = torch.randn(2, 30, 560)           -- fbank [B, T', 560]
  speech_lengths = torch.tensor([6, 30])
  language = torch.tensor([0, 0])            -- 0=auto/default
  textnorm = torch.tensor([15, 15])          -- 15=itn

Output:
  outputs/sensevoice-onnx/model.onnx            ONNX fp32 (fbank input)
  outputs/sensevoice-onnx/fp32_ref_outputs.npz  reference logits
  outputs/sensevoice-onnx/outlier_report.json   outlier scan

Cach chay:
  python src/step1_asr/step4_s1_export_sensevoice_onnx.py
"""
import os
import sys
import json
import time
import glob
import shutil
import traceback
import numpy as np
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")

MODEL_ID_HF = "FunAudioLLM/SenseVoiceSmall"
MODEL_ID_MS = "iic/SenseVoiceSmall"

# Language embedding indices (NOT vocab IDs) -- from export_meta.py dummy + funasr docs
# embed indices used directly in export_forward: language_query = self.embed(language)
LANG_EMBED_IDX = {"auto": 0, "en": 3, "zh": 4, "ko": 7}
TEXTNORM_ITN = 15      # use_itn=True
TEXTNORM_WOITN = 14    # use_itn=False

# Fbank feature dim for SenseVoice-Small (from dummy: torch.randn(2, 30, 560))
FBANK_DIM = 560


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[export] Output dir: {OUT_DIR}")


def load_model():
    """Load SenseVoice via FunASR AutoModel."""
    from funasr import AutoModel
    print("[export] Loading SenseVoice-Small on CPU...")
    t0 = time.time()
    try:
        am = AutoModel(model=MODEL_ID_HF, hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model=MODEL_ID_MS, device="cpu", disable_update=True)
    sv = am.model
    sv.eval()
    print(f"[export] Loaded in {time.time()-t0:.1f}s")
    return am, sv


def apply_export_meta(sv):
    """
    Patch model voi export_forward tu export_meta.py (API chinh thuc FunASR).
    
    export_forward signature:
      forward(speech [B,T',560], speech_lengths [B], language [B], textnorm [B])
    Returns:
      (ctc_logits [B,T',vocab], encoder_out_lens [B])
    """
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    sv_exported = export_rebuild_model(
        sv,
        device="cpu",
        max_seq_len=512,
    )
    print("[export] Applied export_rebuild_model -- forward patched")
    return sv_exported


def extract_fbank(am, wav_path, lang, max_secs=None):
    """
    Extract fbank [1, T', 560] tu raw wav su dung frontend cua model.
    max_secs: neu set, cat ngon audio de tranh OOM khi chay local inference.
    """
    import torch
    import soundfile as sf

    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # Clip audio if needed (local verify only -- avoid O(T^2) OOM)
    if max_secs is not None:
        wav = wav[:int(sr * max_secs)]
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
            
    # Pad to match static shape (1, 500, 560) for ONNX verify
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
        
    return padded_feats, feats_len_np


def export_to_onnx(sv_exported, onnx_path):
    """Export ONNX dung export_dummy_inputs + torch.onnx.export."""
    import torch

    if os.path.exists(onnx_path):
        print(f"[export] Already exists: {onnx_path} -- skipping")
        return True

    print("[export] Getting dummy inputs from export_meta...")
    # OVERRIDE: QAI Hub Hexagon requires static shapes. We fix shape to (1, 500, 560) for max ~29s audio.
    def custom_dummy_inputs(self):
        return (
            torch.randn(1, 500, 560),
            torch.tensor([500], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([15], dtype=torch.int32)
        )
    sv_exported.export_dummy_inputs = types.MethodType(custom_dummy_inputs, sv_exported)
    
    # OVERRIDE: Disable dynamic axes
    def custom_dynamic_axes(self):
        return {}
    sv_exported.export_dynamic_axes = types.MethodType(custom_dynamic_axes, sv_exported)

    dummy = sv_exported.export_dummy_inputs()
    # dummy = (speech [1,500,560], speech_lengths [1], language [1], textnorm [1])
    for i, d in enumerate(dummy):
        print(f"  input[{i}]: shape={d.shape}  dtype={d.dtype}")

    input_names = sv_exported.export_input_names()
    output_names = sv_exported.export_output_names()
    dynamic_axes = sv_exported.export_dynamic_axes()
    print(f"[export] input_names:  {input_names}")
    print(f"[export] output_names: {output_names}")

    # Test forward
    print("[export] Testing forward pass...")
    with torch.no_grad():
        try:
            out = sv_exported(*dummy)
            if isinstance(out, tuple):
                print(f"[export] Forward OK: outputs={[o.shape for o in out]}")
            else:
                print(f"[export] Forward OK: output={out.shape}")
        except Exception as e:
            print(f"[export] Forward FAILED: {e}")
            traceback.print_exc()
            return False

    print("[export] Running torch.onnx.export (30-120s)...")
    t0 = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(
                sv_exported,
                dummy,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
            )
        sz_mb = os.path.getsize(onnx_path) / 1e6
        print(f"[export] ONNX saved: {onnx_path} ({sz_mb:.1f}MB) in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        print(f"[export] torch.onnx.export FAILED: {e}")
        traceback.print_exc()
        return False


def verify_and_save_references(am, onnx_path):
    """
    Verify ONNX chay duoc, luu fp32 reference outputs.
    Input: fbank features (phai extract truoc tu wav).
    """
    import onnxruntime as ort
    import onnx
    import soundfile as sf
    import torch

    print(f"\n[verify] Loading {onnx_path}...")
    model_proto = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(model_proto)
        print(f"[verify] ONNX check OK  ({os.path.getsize(onnx_path)/1e6:.1f}MB)")
    except Exception as e:
        print(f"[verify] ONNX check warning: {e}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_names = [inp.name for inp in sess.get_inputs()]
    out_names = [out.name for out in sess.get_outputs()]
    print(f"[verify] Session inputs:  {inp_names}")
    print(f"[verify] Session outputs: {out_names}")

    with open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    ref_outputs = {}
    for row in manifest:
        if row["lang"] not in ["en", "zh", "ko"]:
            continue
        wav_path = os.path.join(ROOT, row["path"])

        try:
            # Cap audio to 5s to avoid OOM in self-attention O(T^2) on CPU
            MAX_VERIFY_SECS = 5
            fbank, fbank_len = extract_fbank(am, wav_path, row["lang"],
                                             max_secs=MAX_VERIFY_SECS)
            lang_idx = LANG_EMBED_IDX.get(row["lang"], 0)

            feed = {}
            if "speech" in inp_names:
                feed["speech"] = fbank
            if "speech_lengths" in inp_names:
                feed["speech_lengths"] = fbank_len
            if "language" in inp_names:
                feed["language"] = np.array([lang_idx], dtype=np.int32)
            if "textnorm" in inp_names:
                feed["textnorm"] = np.array([TEXTNORM_ITN], dtype=np.int32)

            t0 = time.time()
            outs = sess.run(None, feed)
            elapsed = time.time() - t0

            key = os.path.basename(row["path"])
            ref_outputs[key] = outs[0]  # ctc_logits
            dur = row.get("duration_s", 5.0)
            print(f"[verify] {row['lang']} {key}: logits={outs[0].shape}  RTF={elapsed/dur:.3f}")
        except Exception as e:
            print(f"[verify] WARN {row['path']}: {e}")
            traceback.print_exc()

    ref_path = os.path.join(OUT_DIR, "fp32_ref_outputs.npz")
    np.savez(ref_path, **ref_outputs)
    print(f"[verify] Saved {len(ref_outputs)} references -> {ref_path}")
    return len(ref_outputs) > 0


def inspect_outliers(onnx_path):
    """Scan outlier trong initializers."""
    import onnx
    model = onnx.load(onnx_path)
    outliers = []
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        if arr.size == 0 or arr.dtype not in [np.float32, np.float16]:
            continue
        mn, mx = float(arr.min()), float(arr.max())
        if mn < -1e6 or mx > 1e6:
            outliers.append({"name": init.name, "shape": list(arr.shape), "min": mn, "max": mx})
            print(f"[outlier] {init.name[:60]}  min={mn:.2e}  max={mx:.2e}")

    if not outliers:
        print("[outlier] No static outliers found in initializers")

    report_path = os.path.join(OUT_DIR, "outlier_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(outliers, f, indent=2)
    print(f"[outlier] Report -> {report_path}")
    return outliers


def main():
    print("=" * 60)
    print("Step 4 -- Buoc 1: Export SenseVoice-Small -> ONNX fp32")
    print("Using funasr official export_rebuild_model API")
    print("=" * 60)

    ensure_out_dir()
    onnx_path = os.path.join(OUT_DIR, "model.onnx")

    # Load model
    am, sv = load_model()

    # Apply export_meta patch
    sv_exported = apply_export_meta(sv)

    # Export to ONNX
    success = export_to_onnx(sv_exported, onnx_path)

    if not success or not os.path.exists(onnx_path):
        print("\n[main] FAIL: Could not create model.onnx")
        sys.exit(1)

    # Verify + save references (dung am.model.frontend de extract fbank)
    verify_ok = verify_and_save_references(am, onnx_path)

    # Scan outliers
    outliers = inspect_outliers(onnx_path)

    print("\n" + "=" * 60)
    print("SUMMARY -- Buoc 1/6")
    print("=" * 60)
    print(f"  ONNX export:    {'OK' if success else 'FAIL'}")
    print(f"  Local verify:   {'OK' if verify_ok else 'FAIL'}")
    print(f"  Outliers found: {len(outliers)}")
    print(f"  Output:         {OUT_DIR}")
    print()
    print("NOTE: ONNX input is fbank [B, T', 560], NOT raw waveform!")
    print("      Frontend must be run separately before feeding into ONNX.")
    print()
    print("Next: run step4_s1_patch_mask.py")


if __name__ == "__main__":
    main()
