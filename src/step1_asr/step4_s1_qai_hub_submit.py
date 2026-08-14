"""Step 4 — Bước 4+5/6: Submit Quantize + Compile jobs lên Qualcomm AI Hub

Mục tiêu:
  - Submit quantize job (w8a16: weights=int8, activations=int16) lên AI Hub
  - Submit compile job lên IQ-9075 EVK (Hexagon v73, hỗ trợ w8a16)
  - Lưu job IDs để theo dõi và tiếp tục nếu bị interrupt

CRITICAL — Bài học từ meeting_prep_quantization.md §5.1:
  ✅ ĐÚNG: submit_quantize_job với activations_dtype=INT16
  ❌ SAI:  submit_compile_job với quantize_full_type="int16" (chỉ đổi weights, activation vẫn int8!)
  
  QCS6490/Rubik Pi 3 (Hexagon v68): w8a16 KHÔNG hỗ trợ → FAIL cứng
  IQ-9075 EVK (Hexagon v73): w8a16 ✅ OK — đây là target

Input:  outputs/sensevoice-onnx/model_patched.onnx
        outputs/sensevoice_calib_data.npz
Output: outputs/sensevoice-onnx/qai_job_ids.json   ← lưu job IDs
        outputs/sensevoice-onnx/model_quantized.onnx (sau khi quantize xong)

Cách chạy:
  1. Cài qai-hub:  pip install qai-hub
  2. Auth:         qai-hub configure --api_token YOUR_TOKEN
     (Lấy token tại: https://aihub.qualcomm.com/account)
  3. Chạy script:  python src/step1_asr/step4_s1_qai_hub_submit.py

Để chỉ submit quantize (không compile):
  python src/step1_asr/step4_s1_qai_hub_submit.py --only-quantize

Để resume từ quantize job đã có:
  python src/step1_asr/step4_s1_qai_hub_submit.py --quantize-job-id jXXXXXXXX
"""
import os
import sys
import json
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-onnx")
CALIB_PATH = os.path.join(ROOT, "outputs", "sensevoice_calib_data.npz")
JOB_IDS_PATH = os.path.join(OUT_DIR, "qai_job_ids.json")

# Target device — Hexagon v73+ required for w8a16
# IQ-9075 EVK confirmed working (meeting_prep_quantization.md §6)
TARGET_DEVICE = "Dragonwing IQ-9075 EVK"


def load_job_ids():
    if os.path.exists(JOB_IDS_PATH):
        with open(JOB_IDS_PATH) as f:
            return json.load(f)
    return {}


def save_job_ids(job_ids):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JOB_IDS_PATH, "w") as f:
        json.dump(job_ids, f, indent=2)
    print(f"[qai_hub] Job IDs saved → {JOB_IDS_PATH}")


def check_qai_hub_auth():
    """Kiểm tra qai_hub đã được configure chưa."""
    try:
        import qai_hub as hub
        # Thử list devices — nếu không auth sẽ raise exception
        devices = hub.get_devices()
        print(f"[qai_hub] ✅ Auth OK. Available devices: {len(list(devices))}")
        return True
    except ImportError:
        print("[qai_hub] ❌ qai_hub not installed. Chạy: pip install qai-hub")
        return False
    except Exception as e:
        if "token" in str(e).lower() or "auth" in str(e).lower() or "api" in str(e).lower():
            print(f"[qai_hub] ❌ Auth failed: {e}")
            print("  Cần configure: qai-hub configure --api_token YOUR_TOKEN")
            print("  Lấy token tại: https://aihub.qualcomm.com/account")
        else:
            print(f"[qai_hub] ⚠️  Unexpected error: {e}")
        return False


def prepare_calibration_dataset(calib_npz_path, onnx_path):
    """
    Load calibration data và convert sang format qai_hub DatasetEntries.
    
    qai_hub.DatasetEntries format:
    {
        "input_name": [array_sample_0, array_sample_1, ...]
    }
    """
    import onnxruntime as ort

    print(f"[calib] Loading calibration data from {calib_npz_path}...")
    data = np.load(calib_npz_path, allow_pickle=True)

    # Get input names từ ONNX model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_names = [inp.name for inp in sess.get_inputs()]
    print(f"[calib] ONNX input names: {onnx_input_names}")

    # Reconstruct per-input lists
    dataset = {name: [] for name in onnx_input_names}

    # Đếm số samples
    idx = 0
    while True:
        key = f"{onnx_input_names[0]}_{idx:03d}"
        if key not in data.files:
            break
        for name in onnx_input_names:
            k = f"{name}_{idx:03d}"
            if k in data.files:
                dataset[name].append(data[k])
        idx += 1

    n_samples = idx
    print(f"[calib] Loaded {n_samples} calibration samples")
    for name, arrs in dataset.items():
        print(f"  {name}: {len(arrs)} samples")

    if n_samples == 0:
        raise ValueError("No calibration samples found in NPZ file!")

    return dataset, n_samples


def submit_quantize_job(hub, onnx_path, calib_dataset, job_ids):
    """Submit quantize job với w8a16 (weights=int8, activations=int16)."""
    import qai_hub

    print("\n[quantize] Submitting quantize job (w8a16)...")
    print(f"  weights_dtype:     INT8")
    print(f"  activations_dtype: INT16  ← critical for correctness")
    print(f"  Model:             {onnx_path}")

    # Upload model
    print("[quantize] Uploading model to AI Hub...")
    t0 = time.time()
    model = hub.upload_model(onnx_path)
    print(f"[quantize] Model uploaded in {time.time()-t0:.1f}s")

    # Upload calibration dataset
    print("[quantize] Uploading calibration dataset...")
    t0 = time.time()
    # qai_hub dataset format
    calib_hub = hub.upload_dataset(calib_dataset)
    print(f"[quantize] Dataset uploaded in {time.time()-t0:.1f}s")

    # Submit quantize job
    quantize_job = hub.submit_quantize_job(
        model=model,
        calibration_data=calib_hub,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="SenseVoice-Small_w8a16",
    )

    job_id = quantize_job.job_id
    print(f"[quantize] ✅ Submitted! Job ID: {job_id}")
    print(f"  Track at: https://app.aihub.qualcomm.com/jobs/{job_id}/")

    job_ids["quantize_job_id"] = job_id
    job_ids["quantize_model_id"] = model.model_id if hasattr(model, "model_id") else str(model)
    save_job_ids(job_ids)

    return quantize_job


def wait_for_quantize_job(hub, quantize_job):
    """Đợi quantize job xong và trả về quantized model."""
    print(f"\n[quantize] Waiting for quantize job to complete...")
    print("  (Thường mất 5-20 phút tùy kích thước model)")

    poll_interval = 30  # giây
    max_wait = 3600  # 1 giờ

    start = time.time()
    while time.time() - start < max_wait:
        status = quantize_job.get_status()
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:4d}s] Status: {status}")

        if hasattr(status, "name"):
            status_name = status.name
        else:
            status_name = str(status)

        if "SUCCESS" in status_name.upper() or "DONE" in status_name.upper():
            print(f"[quantize] ✅ Quantize job SUCCEEDED in {elapsed}s")
            return quantize_job.get_target_model()
        elif "FAIL" in status_name.upper() or "ERROR" in status_name.upper():
            print(f"[quantize] ❌ Quantize job FAILED: {status}")
            print("  Xem chi tiết tại AI Hub Workbench UI (API log thường không đủ)")
            sys.exit(1)

        time.sleep(poll_interval)

    print(f"[quantize] ⏱️  Timeout after {max_wait}s. Job ID đã lưu, re-run với --quantize-job-id")
    sys.exit(1)


def submit_compile_job(hub, quantized_model, job_ids):
    """Submit compile job lên IQ-9075 EVK (Hexagon v73)."""
    print(f"\n[compile] Submitting compile job...")
    print(f"  Target device: {TARGET_DEVICE} (Hexagon v73, w8a16 ✅)")
    print("  ⚠️  QCS6490/Rubik Pi 3 (Hexagon v68) sẽ FAIL với w8a16!")

    device = hub.Device(TARGET_DEVICE)

    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        name="SenseVoice-Small_w8a16_IQ9075",
        options="--target_runtime qnn_context_binary",
    )

    job_id = compile_job.job_id
    print(f"[compile] ✅ Submitted! Job ID: {job_id}")
    print(f"  Track at: https://app.aihub.qualcomm.com/jobs/{job_id}/")

    job_ids["compile_job_id"] = job_id
    save_job_ids(job_ids)

    return compile_job


def wait_for_compile_job(hub, compile_job):
    """Đợi compile job xong."""
    print(f"\n[compile] Waiting for compile job to complete...")
    print("  (Thường mất 5-15 phút)")

    poll_interval = 30
    max_wait = 3600
    start = time.time()

    while time.time() - start < max_wait:
        status = compile_job.get_status()
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:4d}s] Status: {status}")

        if hasattr(status, "name"):
            status_name = status.name
        else:
            status_name = str(status)

        if "SUCCESS" in status_name.upper() or "DONE" in status_name.upper():
            print(f"[compile] ✅ Compile job SUCCEEDED in {elapsed}s")
            return compile_job.get_target_model()
        elif "FAIL" in status_name.upper() or "ERROR" in status_name.upper():
            print(f"[compile] ❌ Compile job FAILED: {status}")
            print("\n  Troubleshooting tips:")
            print("  1. Xem log chi tiết tại AI Hub Workbench UI (không phải API)")
            print("  2. Kiểm tra có op nào không hỗ trợ trên HTP không")
            print("  3. Thử simplify model: onnxsim model_patched.onnx model_simplified.onnx")
            sys.exit(1)

        time.sleep(poll_interval)

    print(f"[compile] ⏱️  Timeout. Job ID đã lưu: {compile_job.job_id}")
    sys.exit(1)


def main():
    global TARGET_DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--only-quantize", action="store_true",
                        help="Chỉ submit quantize job, không compile")
    parser.add_argument("--quantize-job-id", type=str,
                        help="Resume từ quantize job đã có (bỏ qua bước quantize)")
    parser.add_argument("--compile-job-id", type=str,
                        help="Resume từ compile job đã có (bỏ qua cả quantize và compile)")
    parser.add_argument("--device", type=str, default=TARGET_DEVICE,
                        help=f"Target device (default: '{TARGET_DEVICE}')")
    args = parser.parse_args()

    TARGET_DEVICE = args.device

    print("=" * 60)
    print("Step 4 — Bước 4+5: Submit Quantize + Compile jobs")
    print("=" * 60)

    # Check qai_hub
    if not check_qai_hub_auth():
        print("\n[setup] Cần cài và configure qai-hub:")
        print("  pip install qai-hub")
        print("  qai-hub configure --api_token YOUR_TOKEN")
        sys.exit(1)

    import qai_hub as hub

    # Load existing job IDs nếu có
    job_ids = load_job_ids()
    if args.quantize_job_id:
        job_ids["quantize_job_id"] = args.quantize_job_id
    if args.compile_job_id:
        job_ids["compile_job_id"] = args.compile_job_id

    # Paths
    onnx_path = os.path.join(OUT_DIR, "model_patched.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(OUT_DIR, "model.onnx")
    if not os.path.exists(onnx_path):
        print("❌ ONNX model not found. Chạy bước 1 và 2 trước.")
        sys.exit(1)
    print(f"[main] ONNX model: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f}MB)")

    # ===========================================================
    # BƯỚC 4: QUANTIZE (w8a16)
    # ===========================================================
    if "compile_job_id" in job_ids and args.compile_job_id:
        # Resume từ compile job
        print(f"\n[main] Resuming from compile job: {job_ids['compile_job_id']}")
        compile_job = hub.get_job(job_ids["compile_job_id"])
        compiled_model = wait_for_compile_job(hub, compile_job)

    elif "quantize_job_id" in job_ids:
        # Resume từ quantize job đã có
        print(f"\n[main] Resuming quantize job: {job_ids['quantize_job_id']}")
        quantize_job = hub.get_job(job_ids["quantize_job_id"])
        quantized_model = wait_for_quantize_job(hub, quantize_job)

        # Lưu quantized model về local
        quant_onnx_path = os.path.join(OUT_DIR, "model_quantized_w8a16.onnx")
        try:
            quantized_model.download(quant_onnx_path)
            print(f"[quantize] Saved quantized model → {quant_onnx_path}")
            job_ids["quantized_onnx"] = quant_onnx_path
            save_job_ids(job_ids)
        except Exception as e:
            print(f"[quantize] ⚠️  Could not download quantized model: {e}")

        if args.only_quantize:
            print("\n✅ Quantize done. Dùng --quantize-job-id để tiếp tục compile.")
            return

        # ===========================================================
        # BƯỚC 5: COMPILE
        # ===========================================================
        compile_job = submit_compile_job(hub, quantized_model, job_ids)
        compiled_model = wait_for_compile_job(hub, compile_job)

    else:
        # Bắt đầu từ đầu
        if not os.path.exists(CALIB_PATH):
            print(f"❌ Calibration data not found at {CALIB_PATH}")
            print("   Chạy step4_s1_prepare_calib.py trước.")
            sys.exit(1)

        calib_dataset, n_samples = prepare_calibration_dataset(CALIB_PATH, onnx_path)
        quantize_job = submit_quantize_job(hub, onnx_path, calib_dataset, job_ids)
        quantized_model = wait_for_quantize_job(hub, quantize_job)

        # Download quantized model
        quant_onnx_path = os.path.join(OUT_DIR, "model_quantized_w8a16.onnx")
        try:
            quantized_model.download(quant_onnx_path)
            print(f"[quantize] Saved quantized model → {quant_onnx_path}")
            job_ids["quantized_onnx"] = quant_onnx_path
            save_job_ids(job_ids)
        except Exception as e:
            print(f"[quantize] ⚠️  Could not download: {e}")

        if args.only_quantize:
            print("\n✅ Quantize done. Chạy lại với --quantize-job-id để compile.")
            return

        compile_job = submit_compile_job(hub, quantized_model, job_ids)
        compiled_model = wait_for_compile_job(hub, compile_job)

    # Lưu compiled model ID
    if hasattr(compiled_model, "model_id"):
        job_ids["compiled_model_id"] = compiled_model.model_id
    save_job_ids(job_ids)

    print("\n" + "=" * 60)
    print("SUMMARY — Bước 4+5/6")
    print("=" * 60)
    print(f"  Quantize job ID:  {job_ids.get('quantize_job_id', 'N/A')}")
    print(f"  Compile job ID:   {job_ids.get('compile_job_id', 'N/A')}")
    print(f"  Compiled model:   {job_ids.get('compiled_model_id', 'N/A')}")
    print(f"  Job IDs file:     {JOB_IDS_PATH}")
    print()
    print("Bước tiếp theo: chạy step4_s1_verify_w8a16.py để verify cosine similarity + WER/CER")


if __name__ == "__main__":
    main()
