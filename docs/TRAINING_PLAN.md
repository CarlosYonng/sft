# Training Plan

## Goal

Use Qwen2.5-3B-Instruct as the base model and Medical-R1-Distill-Data-Chinese as the first vertical medical SFT dataset. The first production of this repository is a LoRA adapter that is small enough to train on RTX 4070-class hardware and easy to merge or reuse.

The default configuration trains on final answers only and does not expose the dataset reasoning field. This is more suitable for a medical assistant because it reduces verbose chain-of-thought style output and makes safety review easier.

## Phases

1. Smoke run
   - Samples: 128-512
   - Steps: 20-80
   - Purpose: verify data format, tokenizer chat template, GPU memory, save/load path, and inference.

2. Controlled run
   - Samples: 2k-5k
   - Epochs: 1-2
   - Purpose: get a visible medical-domain style shift within a manageable training time.

3. Full dataset run
   - Samples: all 17k
   - Epochs: 2-3
   - Purpose: create the real adapter after hyperparameters are confirmed.

## Default RTX 4070 Settings

- Method: LoRA fp16 by default, QLoRA when bitsandbytes works
- Sequence length: 1024 for first run, then 2048
- Batch size: 1
- Gradient accumulation: 8
- LoRA rank: 16 first, then 32 if memory allows
- Learning rate: 1e-4 for LoRA, 2e-4 for QLoRA

## Validation

The model must be checked with:

- Emergency medicine questions
- Drug safety questions
- Pediatric and pregnancy contraindication questions
- Insufficient-information cases
- Non-medical or unsafe requests
- Comparison against the base model on the same prompt set

Do not use this model for real diagnosis without clinical review and stronger safety evaluation.
