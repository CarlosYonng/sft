from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.compare_models import generate_answer
from sft_toolkit.config import load_config


CONFIG_PATH = "configs/medical_qwen25_3b.yaml"
QUESTION = "糖尿病患者空腹血糖长期偏高，应该如何处理？"
MAX_NEW_TOKENS = 512


def main() -> None:
    config = load_config(CONFIG_PATH)
    base_model = config["paths"]["base_model"]
    adapter = "outputs/qwen25-3b-medical-lora"

    print("\nQUESTION")
    print(QUESTION)

    print("\n" + "=" * 24 + " BASE MODEL " + "=" * 24)
    base_answer = generate_answer(base_model, config, QUESTION, MAX_NEW_TOKENS)
    print(base_answer)

    print("\n" + "=" * 20 + " FINE-TUNED LORA " + "=" * 20)
    tuned_answer = generate_answer(base_model, config, QUESTION, MAX_NEW_TOKENS, adapter)
    print(tuned_answer)


if __name__ == "__main__":
    main()
