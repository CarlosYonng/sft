from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.compare_models import generate_answer
from sft_toolkit.config import load_config


CONFIG_PATH = "configs/medical_qwen25_3b.yaml"
QUESTION = "儿童高热惊厥时家属应该怎么做？"
MAX_NEW_TOKENS = 512


def main() -> None:
    config = load_config(CONFIG_PATH)
    base_model = config["paths"]["base_model"]
    answer = generate_answer(base_model, config, QUESTION, MAX_NEW_TOKENS)
    print("\nQUESTION")
    print(QUESTION)
    print("\nBASE MODEL ANSWER")
    print(answer)


if __name__ == "__main__":
    main()
