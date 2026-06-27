from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from sft_toolkit.config import load_config
from sft_toolkit.modeling import load_tokenizer


DEFAULT_QUESTIONS = [
    "患者男，58岁，胸痛30分钟、大汗，心电图提示ST段抬高，应如何紧急处理？",
    "儿童高热惊厥时家属应该怎么做？哪些情况必须立即就医？",
    "孕妇感冒发热时可以自行服用抗生素吗？需要注意什么？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat test a base or LoRA model.")
    parser.add_argument("--config", default="configs/medical_qwen25_3b.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = args.model or config["paths"]["base_model"]
    tokenizer = load_tokenizer(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for question in DEFAULT_QUESTIONS:
        messages = [
            {"role": "system", "content": config["data"]["system_prompt"]},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                temperature=float(config["generation"]["temperature"]),
                top_p=float(config["generation"]["top_p"]),
                repetition_penalty=float(config["generation"]["repetition_penalty"]),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        print("\nQ:", question)
        print("A:", answer.strip())


if __name__ == "__main__":
    main()

