from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from sft_toolkit.config import load_config
from sft_toolkit.modeling import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument("--config", default="configs/medical_qwen25_3b.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter = args.adapter or config["paths"]["output_dir"]
    out = args.out or config["paths"]["merged_dir"]

    tokenizer = load_tokenizer(config["paths"]["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["paths"]["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(out)
    print(f"Merged model saved to: {out}")


if __name__ == "__main__":
    main()

