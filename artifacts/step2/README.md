# Step 2 MT artifact

Artifact chính thức: `OneVoice_MT_FINAL_QUALITY.zip`

- Kích thước: 32.340.857 bytes (30,84 MiB)
- SHA-256: `D20D47F15813907F6087EDA510E8B9659DEAB00B623A32B82FE43D30539170E5`
- ZIP members: 124
- ZIP integrity: PASS
- Benchmark baseline: 71.796 lượt dịch
- NLLB-1.3B comparison: 12.144 lượt FLORES
- Streaming/AlignAtt-text: 12/12 hướng

Nội dung ZIP gồm code Step 2, báo cáo, manifest dataset, summary/detail benchmark, COMET, GPU profile, fine-tune comparison và local INT8 validation. Model binaries và PhoMT thô không được đóng gói.

Kiểm tra checksum trên PowerShell:

```powershell
Get-FileHash artifacts/step2/OneVoice_MT_FINAL_QUALITY.zip -Algorithm SHA256
```

Kiểm tra ZIP bằng Python:

```bash
python -c "import zipfile; print(zipfile.ZipFile('artifacts/step2/OneVoice_MT_FINAL_QUALITY.zip').testzip())"
```

Kết quả mong đợi là `None`.
