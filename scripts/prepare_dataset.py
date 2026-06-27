from __future__ import annotations

import argparse
from pathlib import Path

from sft_toolkit.config import load_config
from sft_toolkit.data import load_raw_json, split_examples, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare medical SFT dataset as JSONL splits.")
    parser.add_argument("--config", default="configs/medical_qwen25_3b.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = load_raw_json(config["paths"]["raw_dataset"])
    train, eval_rows = split_examples(
        rows=rows,
        system_prompt=config["data"]["system_prompt"],
        include_reasoning=bool(config["data"]["include_reasoning"]),
        eval_ratio=float(config["data"]["eval_ratio"]),
        seed=int(config["data"]["seed"]),
    )
    train_count = write_jsonl(config["paths"]["processed_train"], train)
    eval_count = write_jsonl(config["paths"]["processed_eval"], eval_rows)
    print(f"Prepared train={train_count} eval={eval_count}")
    print(f"Train file: {Path(config['paths']['processed_train']).resolve()}")
    print(f"Eval file: {Path(config['paths']['processed_eval']).resolve()}")


if __name__ == "__main__":
    main()

