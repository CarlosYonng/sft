# Run Record

Date: 2026-06-27

## Environment

- OS shell: Windows PowerShell
- Python env: `.conda/sft-med`
- GPU detected by `nvidia-smi`: NVIDIA GeForce RTX 4070
- GPU memory detected: about 12GB
- Torch: 2.5.1+cu121
- Transformers: 5.12.1
- PEFT: 0.19.1
- Accelerate: 1.14.0

## Data

- Raw dataset: `E:/sft/models/Medical-R1-Distill-Data-Chinese/medical_r1_distill_sft_Chinese.json`
- Raw rows: 17,000
- Processed train rows: 16,490
- Processed eval rows: 510
- Final training mode: response-only, without explicit reasoning field

## Training

Final controlled run:

- Base model: `E:/sft/models/Qwen2.5-3B-Instruct`
- Adapter output: `outputs/qwen25-3b-medical-lora`
- Merged model output: `outputs/qwen25-3b-medical-merged`
- Samples used: 512
- Max steps: 60
- LoRA rank: 16
- Trainable parameters: 29,933,568
- Trainable ratio: 0.9607%
- Max sequence length: 1024
- Batch size: 1
- Gradient accumulation: 8
- Runtime: about 9 minutes 18 seconds
- Final train loss: 1.108
- Final eval loss: 1.09

An earlier reasoning-enabled run was discarded as the recommended default because it produced verbose `<think>` output.

## Validation Summary

Passed:

- Adapter loads successfully.
- Merged model loads successfully.
- Response-only training removed visible `<think>` output in normal chat tests.
- The fixed unsafe sleep-medication prompt no longer returned operational overdose steps after strengthening the system prompt.
- The model shows an obvious Chinese medical answer style after a short controlled run.

Known risks:

- Some emergency medicine answers still include overly specific or inaccurate treatment details.
- Some pediatric and pregnancy medication answers can be too permissive or imprecise.
- This quick run is an experiment baseline, not a clinically validated model.

Recommended next run:

- Keep response-only training.
- Increase data to 2,000-5,000 samples.
- Add a medical safety/refusal dataset.
- Add a doctor-reviewed validation set for emergency, pregnancy, pediatric, and drug dosage scenarios.
- Consider filtering training outputs that contain concrete drug dosage or procedure details before the next run.
