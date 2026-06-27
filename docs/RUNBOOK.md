# Runbook

## Environment

Recommended:

```powershell
conda create -n sft-med python=3.10 -y
conda activate sft-med
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Optional QLoRA:

```powershell
pip install -r requirements-qlora.txt
```

## Prepare Dataset

```powershell
python scripts/prepare_dataset.py --config configs/medical_qwen25_3b.yaml
```

## Train

Short controlled run:

```powershell
python scripts/train_sft.py --config configs/medical_qwen25_3b.yaml --max-samples 512 --max-steps 60
```

Full run:

```powershell
python scripts/train_sft.py --config configs/medical_qwen25_3b.yaml
```

## Test

```powershell
python scripts/chat_test.py --config configs/medical_qwen25_3b.yaml --adapter outputs/qwen25-3b-medical-lora
python scripts/evaluate_prompts.py --config configs/medical_qwen25_3b.yaml --adapter outputs/qwen25-3b-medical-lora
```

## Merge

```powershell
python scripts/merge_lora.py --config configs/medical_qwen25_3b.yaml --adapter outputs/qwen25-3b-medical-lora
```
