from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.compare_models import generate_answer
from sft_toolkit.config import load_config


CONFIG_PATH = "configs/medical_qwen25_3b.yaml"
QUESTION = "孕妇感冒发热时可以自行服用抗生素吗？需要注意什么？"
ADAPTER_PATH = "outputs/qwen25-3b-medical-lora"
MAX_NEW_TOKENS = 512


def main() -> None:
    config = load_config(CONFIG_PATH)
    base_model = config["paths"]["base_model"]
    answer = generate_answer(base_model, config, QUESTION, MAX_NEW_TOKENS, ADAPTER_PATH)
    print("\nQUESTION")
    print(QUESTION)
    print("\nLORA MODEL ANSWER")
    print(answer)


if __name__ == "__main__":
    main()
