# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import h5py

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\ChuyenNganhAI\AuraTranslateEdge-OneVoice"
DEFAULT_H5 = os.path.join(ROOT, "outputs", "dataset-d9505wgw7.h5")

SP_PATHS = [
    os.path.expanduser(r"~/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/chn_jpn_yue_eng_ko_spectok.bpe.model"),
    os.path.expanduser(r"~/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master/chn_jpn_yue_eng_ko_spectok.bpe.model"),
]

def load_tokenizer():
    import sentencepiece as spm
    for p in SP_PATHS:
        if os.path.exists(p):
            sp = spm.SentencePieceProcessor()
            sp.load(p)
            return sp, p
    raise FileNotFoundError("Không tìm thấy file chn_jpn_yue_eng_ko_spectok.bpe.model!")

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
    parser = argparse.ArgumentParser(description="Decode Qualcomm AI Hub H5 inference dataset")
    parser.add_argument("h5_path", nargs="?", default=DEFAULT_H5, help="Path to .h5 file")
    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"[ERROR] Không tìm thấy file H5 tại: {args.h5_path}")
        return

    sp, sp_model_path = load_tokenizer()
    print(f"Loaded Tokenizer: {sp_model_path}")
    print(f"Reading H5 file : {args.h5_path}\n")

    LANG_NAMES = ["English (En)", "Chinese (Zh)", "Korean (Ko)"]

    with h5py.File(args.h5_path, "r") as f:
        batches = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                batches.append((name, obj))
        f.visititems(visitor)

        batches.sort(key=lambda x: x[0])
        print("=" * 70)
        print("KẾT QUẢ GIẢI MÃ TỪ PHẦN CỨNG QUALCOMM DRAGONWING IQ-9075 EVK (NPU)")
        print("=" * 70)

        for i, (name, ds) in enumerate(batches):
            tokens = ds[:]
            lang_label = LANG_NAMES[i] if i < len(LANG_NAMES) else f"Sample {i}"
            decoded_text, clean_tokens = ctc_decode(tokens, sp)
            print(f"\n[{lang_label}] — Node: {name} (Shape: {ds.shape}, Dtype: {ds.dtype})")
            print(f"  Raw Token IDs (first 15): {clean_tokens[:15]}")
            print(f"  Decoded String          : {decoded_text}")

if __name__ == "__main__":
    main()
