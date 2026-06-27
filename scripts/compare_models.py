from __future__ import annotations

import argparse
import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from sft_toolkit.config import load_config
from sft_toolkit.modeling import load_tokenizer


DEFAULT_QUESTION = "患者男，58岁，胸痛30分钟、大汗，心电图提示ST段抬高，应如何紧急处理？"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base model and fine-tuned model answers.")
    parser.add_argument("--config", default="configs/medical_qwen25_3b.yaml")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--adapter", default="outputs/qwen25-3b-medical-lora")
    parser.add_argument("--merged-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def unload_model(model: AutoModelForCausalLM | PeftModel | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_answer(model_path: str, config: dict, question: str, max_new_tokens: int, adapter: str | None = None) -> str:
    tokenizer = load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    messages = [
        {"role": "system", "content": config["data"]["system_prompt"]},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(config["generation"]["temperature"]),
            top_p=float(config["generation"]["top_p"]),
            repetition_penalty=float(config["generation"]["repetition_penalty"]),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    unload_model(model)
    return answer.strip()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_model = config["paths"]["base_model"]
    tuned_model = args.merged_model or base_model
    tuned_adapter = None if args.merged_model else args.adapter

    print("\nQUESTION")
    print(args.question)

    print("\n" + "=" * 24 + " BASE MODEL " + "=" * 24)
    print(generate_answer(base_model, config, args.question, args.max_new_tokens))

    print("\n" + "=" * 20 + " FINE-TUNED MODEL " + "=" * 20)
    print(generate_answer(tuned_model, config, args.question, args.max_new_tokens, tuned_adapter))


if __name__ == "__main__":
    main()
