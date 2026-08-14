import numpy as np
import qai_hub as hub
import os
import json

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    if a.shape != b.shape:
        raise ValueError(f"shapes {a.shape} and {b.shape} not aligned")
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

job_ids = [
    "jp36yexmp", "jpym96q0p", "jp2e2lv6p", "jpxxwyx8p", "jg9d6wmq5",
    "jgk8j36yg", "jpeyjkx75", "jgo8j394p", "jpym96k4p", "jp36yedzp",
    "jpym96e4p", "j56wk126g", "jg9d64885", "jgk8jy2vg", "jgjqjl985"
]

manifest = json.load(open("data/asr/manifest.json", "r", encoding="utf-8"))
samples = [r for r in manifest if r["lang"] in ["en", "zh", "ko"]]

fp32_refs = np.load("outputs/sensevoice-onnx/fp32_ref_outputs.npz", allow_pickle=True)

results = []

for idx, job_id in enumerate(job_ids):
    row = samples[idx]
    key = os.path.basename(row["path"])
    print(f"Processing {key} from job {job_id}...")
    
    fp32_out = fp32_refs[key]  # shape [1, T_fp32, 25055]
    
    # Download hw_out
    job = hub.get_job(job_id)
    out_dict = job.download_output_data()
    hw_outputs = out_dict
    if "0" in out_dict and isinstance(out_dict["0"], dict):
        hw_outputs = out_dict["0"]
    if "output_0" in hw_outputs:
        hw_out = hw_outputs["output_0"][0]
    else:
        hw_out = list(hw_outputs.values())[0][0]
    
    hw_out = np.array(hw_out) # shape [1, 500, 25055]
    
    # Crop hw_out to match fp32_out sequence length
    seq_len = fp32_out.shape[1]
    
    # SenseVoice returns [1, T, 25055]
    if hw_out.ndim == 3 and fp32_out.ndim == 3:
        hw_out_cropped = hw_out[:, :seq_len, :]
    elif hw_out.ndim == 2 and fp32_out.ndim == 2:
        hw_out_cropped = hw_out[:seq_len, :]
    else:
        print(f"Shapes {hw_out.shape} and {fp32_out.shape} don't match logic")
        hw_out_cropped = hw_out

    try:
        cos_sim = cosine_similarity(fp32_out, hw_out_cropped)
        print(f"{key}: cos_sim = {cos_sim:.4f}")
        results.append(cos_sim)
    except Exception as e:
        print(f"Failed {key}: {e}")

print("Average Cosine Similarity:", np.mean(results))
