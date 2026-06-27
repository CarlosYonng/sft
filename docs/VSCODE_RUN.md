# VSCode Run Guide

本项目已内置 `.vscode/launch.json`，可以直接用 VSCode 的运行面板启动模型测试，不需要手动输入命令。

## 使用步骤

1. 在 VSCode 打开项目目录：`C:\Users\CarlorYonng\Documents\sft`
2. 点击左侧的 **运行和调试** 图标。
3. 顶部下拉框选择一个配置。
4. 点击绿色三角按钮运行。

## 常用配置

- `SFT: Compare Base vs LoRA`
  - 输入一个问题。
  - 自动先跑基座模型，再跑 LoRA 微调模型。
  - 最适合观察微调效果。

- `SFT: Chat Base Model`
  - 只跑微调前的 `Qwen2.5-3B-Instruct`。

- `SFT: Chat LoRA Model`
  - 跑 `基座模型 + outputs/qwen25-3b-medical-lora`。

- `SFT: Chat Merged Model`
  - 跑 `outputs/qwen25-3b-medical-merged`。

- `SFT: Evaluate LoRA`
  - 跑固定医疗安全评估问题。
  - 输出保存到 `data/reports/vscode_prompt_eval.jsonl`。

## 注意

第一次加载模型会比较慢。运行前建议关闭占用 GPU 的其他程序。
