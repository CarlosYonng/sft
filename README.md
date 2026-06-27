# Medical SFT Toolkit

面向中文医疗领域的 Qwen SFT 微调项目。当前默认场景使用：

- Base model: `E:/sft/models/Qwen2.5-3B-Instruct`
- Dataset: `E:/sft/models/Medical-R1-Distill-Data-Chinese/medical_r1_distill_sft_Chinese.json`
- Recommended GPU: RTX 4070 12GB/16GB or better

项目按可复用方式组织，后续可以继续加入其他模型、数据集和训练配置。

## Project Layout

```text
.
├── configs/                 # 训练、评估、LLaMA-Factory 配置
├── data/                    # 只放处理后的轻量索引/样例，不提交原始大数据
├── docs/                    # 训练计划、验证方案、运行记录
├── scripts/                 # 命令行入口
├── src/sft_toolkit/         # 可复用代码
└── tests/                   # 轻量测试
```

## Quick Start

创建环境后执行一次小规模训练：

```powershell
python scripts/prepare_dataset.py --config configs/medical_qwen25_3b.yaml
python scripts/train_sft.py --config configs/medical_qwen25_3b.yaml --max-samples 256 --max-steps 30
python scripts/chat_test.py --config configs/medical_qwen25_3b.yaml --adapter outputs/qwen25-3b-medical-lora
```

完整训练建议：

```powershell
python scripts/train_sft.py --config configs/medical_qwen25_3b.yaml
python scripts/merge_lora.py --config configs/medical_qwen25_3b.yaml --adapter outputs/qwen25-3b-medical-lora
python scripts/evaluate_prompts.py --config configs/medical_qwen25_3b.yaml --model outputs/qwen25-3b-medical-merged
```

## VSCode Direct Run

如果不想手动输入命令，可以直接打开根目录的 `run_compare.py`、`run_base.py`、`run_lora.py` 或 `run_merged.py`，点击 VSCode 右上角 **Run Python File**，也可以打断点调试。

详细说明见 [docs/VSCODE_RUN.md](docs/VSCODE_RUN.md)。

## Hardware Notes

RTX 4070 12GB 建议首轮配置：

- LoRA fp16 when bitsandbytes is unavailable
- `max_seq_length: 1024`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 8`
- `lora_rank: 16` or `32`

如果安装了可用的 bitsandbytes，可切换到 4-bit QLoRA，并把序列长度逐步提高到 2048。

## Medical Safety

本项目生成的是研究/实验模型，不可直接作为医疗诊断系统上线。验证时必须覆盖急症、药物安全、孕产妇/儿童禁忌、信息不足拒答和幻觉测试。

默认配置不训练显式 `reasoning` 字段，只训练最终回答，避免医疗助手在实际对话中输出大段内部推理。
