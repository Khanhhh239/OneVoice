# -*- coding: utf-8 -*-
import os
import sys
import glob
import argparse
import numpy as np
import h5py

sys.stdout.reconfigure(encoding='utf-8')

# Tu dong nhan dien thu muc goc cua repository
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# Thu muc chua outputs
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
VERIFIED_H5 = os.path.join(DEFAULT_OUTPUT_DIR, "dataset-d7m8ojpl2.h5")

SP_PATHS = [
    os.path.expanduser(r"~/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/chn_jpn_yue_eng_ko_spectok.bpe.model"),
    os.path.expanduser(r"~/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master/chn_jpn_yue_eng_ko_spectok.bpe.model"),
]

def find_default_h5():
    # 1. Uu tien file dataset chuan da nghiem thu 100% (dataset-d7m8ojpl2.h5)
    if os.path.exists(VERIFIED_H5):
        return VERIFIED_H5
    
    # 2. Tim bat ky file .h5 nao trong outputs/sensevoice-e2e-onnx
    h5_candidates = glob.glob(os.path.join(DEFAULT_OUTPUT_DIR, "*.h5"))
    if h5_candidates:
        h5_candidates.sort(key=os.path.getmtime, reverse=True)
        return h5_candidates[0]
        
    # 3. Tim trong outputs/
    h5_candidates_root = glob.glob(os.path.join(ROOT, "outputs", "*.h5"))
    if h5_candidates_root:
        h5_candidates_root.sort(key=os.path.getmtime, reverse=True)
        return h5_candidates_root[0]
        
    return VERIFIED_H5

def load_tokenizer():
    import sentencepiece as spm
    for p in SP_PATHS:
        if os.path.exists(p):
            sp = spm.SentencePieceProcessor()
            sp.load(p)
            return sp, p
    raise FileNotFoundError("Khong tim thay file tokenizer chn_jpn_yue_eng_ko_spectok.bpe.model trong cache HuggingFace/ModelScope!")

def ctc_decode(token_ids, sp):
    token_ids = np.array(token_ids).flatten()
    blank_id = 0
    collapsed = []
    prev = -1
    for t in token_ids:
        if t != blank_id and t != prev:
            collapsed.append(int(t))
        prev = t
    return sp.decode(collapsed), collapsed

def main():
    default_h5 = find_default_h5()
    parser = argparse.ArgumentParser(description="Giai ma ket qua dataset .h5 tu Qualcomm AI Hub NPU")
    parser.add_argument("h5_path", nargs="?", default=default_h5, help="Duong dan toi file .h5 (mac dinh tu dong tim file moi nhat)")
    args = parser.parse_args()

    target_h5 = os.path.abspath(args.h5_path)
    if not os.path.exists(target_h5):
        print(f"[ERROR] Khong tim thay file H5 tai: {target_h5}")
        print(f"Goi y: Dat file .h5 vao thu muc: {DEFAULT_OUTPUT_DIR}")
        return

    sp, sp_model_path = load_tokenizer()
    print(f"Loaded Tokenizer: {sp_model_path}")
    print(f"Reading H5 file : {target_h5}\n")

    LANG_NAMES = ["English (En)", "Chinese (Zh)", "Korean (Ko)"]

    with h5py.File(target_h5, "r") as f:
        batches = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                batches.append((name, obj))
        f.visititems(visitor)

        batches.sort(key=lambda x: x[0])
        print("=" * 75)
        print("KET QUA GIAI MA TU PHAN CUNG QUALCOMM DRAGONWING IQ-9075 EVK (NPU)")
        print("=" * 75)

        for i, (name, ds) in enumerate(batches):
            tokens = ds[:]
            lang_label = LANG_NAMES[i] if i < len(LANG_NAMES) else f"Sample {i}"
            decoded_text, clean_tokens = ctc_decode(tokens, sp)
            print(f"\n[{lang_label}] -> Node: {name} (Shape: {ds.shape}, Dtype: {ds.dtype})")
            print(f"  Raw Token IDs (first 15): {clean_tokens[:15]}")
            print(f"  Decoded String          : {decoded_text}")

if __name__ == "__main__":
    main()
