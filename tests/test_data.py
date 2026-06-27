from sft_toolkit.data import normalize_row


def test_normalize_medical_r1_columns() -> None:
    row = {
        "question": "问题",
        "reasoning (reasoning_content)": "推理",
        "response (content)": "回答",
    }
    example = normalize_row(row, "系统", include_reasoning=True)
    assert example.system == "系统"
    assert example.question == "问题"
    assert "<think>" in example.answer
    assert "回答" in example.answer
