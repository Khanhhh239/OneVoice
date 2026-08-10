"""LoRA fine-tuning for NLLB from licensed JSONL parallel data.

Input rows: {"source": str, "reference": str, "src_lang": "vi", "tgt_lang": "en"}.
FLORES/NTREX benchmark files must never be passed as training data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)

try:
    from .engine import LANG_CODES
except ImportError:
    from engine import LANG_CODES


def load_parallel(path: Path, src: str, tgt: str, limit: int | None) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("src_lang", src) == src and row.get("tgt_lang", tgt) == tgt:
                rows.append({"source": row["source"], "reference": row["reference"]})
                if limit and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"No {src}->{tgt} examples in {path}")
    return Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path)
    p.add_argument("--src", choices=LANG_CODES, required=True)
    p.add_argument("--tgt", choices=LANG_CODES, required=True)
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.src == args.tgt:
        raise ValueError("Source and target must differ")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, src_lang=LANG_CODES[args.src], tgt_lang=LANG_CODES[args.tgt])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()

    def tokenize(batch):
        encoded = tokenizer(batch["source"], max_length=256, truncation=True)
        labels = tokenizer(text_target=batch["reference"], max_length=256, truncation=True)
        encoded["labels"] = labels["input_ids"]
        return encoded

    train = load_parallel(args.train, args.src, args.tgt, args.limit).map(
        tokenize, batched=True, remove_columns=["source", "reference"])
    valid = None
    if args.validation:
        valid = load_parallel(args.validation, args.src, args.tgt, 1000).map(
            tokenize, batched=True, remove_columns=["source", "reference"])

    training = Seq2SeqTrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, warmup_ratio=0.03,
        fp16=torch.cuda.is_available(), logging_steps=25,
        save_strategy="epoch", eval_strategy="epoch" if valid else "no",
        predict_with_generate=False, report_to="none", seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training, train_dataset=train, eval_dataset=valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "training_manifest.json").write_text(json.dumps({
        "base_model": args.model, "train_file": str(args.train),
        "src": args.src, "tgt": args.tgt, "examples": len(train),
        "epochs": args.epochs, "seed": args.seed,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
