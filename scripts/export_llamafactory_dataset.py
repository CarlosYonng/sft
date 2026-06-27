from __future__ import annotations

import argparse
import json
from pathlib import Path

from sft_toolkit.config import load_config
from sft_toolkit.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed JSONL to LLaMA-Factory alpaca JSON.")
    parser.add_argument("--config", default="configs/medical_qwen25_3b.yaml")
    parser.add_argument("--out-dir", default="data/llamafactory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = read_jsonl(config["paths"]["processed_train"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "medical_r1_zh.json"
    lf_rows = [
        {
            "instruction": row["question"],
            "input": "",
            "output": row["answer"],
            "system": row["system"],
        }
        for row in rows
    ]
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(lf_rows, f, ensure_ascii=False, indent=2)

    dataset_info = {
        "medical_r1_zh": {
            "file_name": "medical_r1_zh.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
    }
    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(lf_rows)} rows to {out_file.resolve()}")


if __name__ == "__main__":
    main()

