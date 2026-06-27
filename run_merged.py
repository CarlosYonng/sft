from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.compare_models import generate_answer
from sft_toolkit.config import load_config


CONFIG_PATH = "configs/medical_qwen25_3b.yaml"
QUESTION = "患者男，58岁，胸痛30分钟、大汗，心电图提示ST段抬高，应如何紧急处理？"
MERGED_MODEL_PATH = "outputs/qwen25-3b-medical-merged"
MAX_NEW_TOKENS = 512


def main() -> None:
    config = load_config(CONFIG_PATH)
    answer = generate_answer(MERGED_MODEL_PATH, config, QUESTION, MAX_NEW_TOKENS)
    print("\nQUESTION")
    print(QUESTION)
    print("\nMERGED MODEL ANSWER")
    print(answer)


if __name__ == "__main__":
    main()
