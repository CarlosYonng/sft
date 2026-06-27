from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


QUESTION_KEYS = ("question", "instruction", "prompt")
REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning (reasoning_content)")
RESPONSE_KEYS = ("response", "content", "output", "response (content)")


@dataclass(frozen=True)
class SftExample:
    system: str
    question: str
    answer: str


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_row(row: dict[str, Any], system_prompt: str, include_reasoning: bool) -> SftExample:
    question = _first_present(row, QUESTION_KEYS)
    reasoning = _first_present(row, REASONING_KEYS)
    response = _first_present(row, RESPONSE_KEYS)
    if not question or not response:
        raise ValueError(f"Bad row without question/response keys: {row.keys()}")

    if include_reasoning and reasoning:
        answer = f"<think>\n{reasoning}\n</think>\n\n{response}"
    else:
        answer = response

    return SftExample(system=system_prompt, question=question, answer=answer)


def load_raw_json(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("train", "data", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or dict with train/data/rows: {source}")
    return data


def write_jsonl(path: str | Path, examples: Iterable[SftExample]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.__dict__, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_examples(
    rows: list[dict[str, Any]],
    system_prompt: str,
    include_reasoning: bool,
    eval_ratio: float,
    seed: int,
) -> tuple[list[SftExample], list[SftExample]]:
    examples = [normalize_row(row, system_prompt, include_reasoning) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(examples)
    eval_size = max(1, int(len(examples) * eval_ratio))
    return examples[eval_size:], examples[:eval_size]

