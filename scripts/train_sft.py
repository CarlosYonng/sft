from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

from sft_toolkit.config import load_config
from sft_toolkit.data import read_jsonl
from sft_toolkit.modeling import attach_lora, load_base_model, load_tokenizer
from sft_toolkit.torch_dataset import ChatSftDataset, SftDataCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen medical LoRA adapter.")
    parser.add_argument("--config", default="configs/medical_qwen25_3b.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def training_args_kwargs(raw: dict, output_dir: str, max_steps: int | None) -> dict:
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(raw["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(raw["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(raw["gradient_accumulation_steps"]),
        "learning_rate": float(raw["learning_rate"]),
        "num_train_epochs": float(raw["num_train_epochs"]),
        "max_steps": int(max_steps if max_steps is not None else raw["max_steps"]),
        "warmup_ratio": float(raw["warmup_ratio"]),
        "weight_decay": float(raw["weight_decay"]),
        "logging_steps": int(raw["logging_steps"]),
        "save_steps": int(raw["save_steps"]),
        "eval_steps": int(raw["eval_steps"]),
        "save_total_limit": int(raw["save_total_limit"]),
        "fp16": bool(raw["fp16"]),
        "bf16": bool(raw["bf16"]),
        "gradient_checkpointing": bool(raw["gradient_checkpointing"]),
        "dataloader_num_workers": int(raw["dataloader_num_workers"]),
        "report_to": "none",
        "remove_unused_columns": False,
        "lr_scheduler_type": "cosine",
        "do_train": True,
        "do_eval": True,
        "save_strategy": "steps",
        "logging_strategy": "steps",
    }

    # Transformers renamed this argument; keep the script usable across versions.
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["data"]["seed"]))

    train_file = Path(config["paths"]["processed_train"])
    eval_file = Path(config["paths"]["processed_eval"])
    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError("Run scripts/prepare_dataset.py before training.")

    train_rows = read_jsonl(train_file)
    eval_rows = read_jsonl(eval_file)
    if args.max_samples:
        train_rows = train_rows[: args.max_samples]
        eval_rows = eval_rows[: max(8, min(len(eval_rows), args.max_samples // 10))]

    tokenizer = load_tokenizer(config["paths"]["base_model"])
    train_dataset = ChatSftDataset(train_rows, tokenizer, int(config["training"]["max_seq_length"]))
    eval_dataset = ChatSftDataset(eval_rows, tokenizer, int(config["training"]["max_seq_length"]))
    if len(train_dataset) == 0:
        raise ValueError("No trainable samples after tokenization/truncation.")

    use_4bit = bool(config["training"]["use_4bit"])
    model = load_base_model(
        config["paths"]["base_model"],
        use_4bit=use_4bit,
        fp16=bool(config["training"]["fp16"]),
        bf16=bool(config["training"]["bf16"]),
    )
    model = attach_lora(model, config["lora"], use_4bit=use_4bit)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            **training_args_kwargs(config["training"], config["paths"]["output_dir"], args.max_steps)
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SftDataCollator(tokenizer),
    )

    result = trainer.train()
    trainer.save_model(config["paths"]["output_dir"])
    tokenizer.save_pretrained(config["paths"]["output_dir"])

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_result.json").open("w", encoding="utf-8") as f:
        json.dump(result.metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
