# VSCode Run Guide

本项目支持像普通 Python 开发一样直接运行文件、打断点、看集成终端输出。

## 使用步骤

1. 在 VSCode 打开项目目录：`C:\Users\CarlorYonng\Documents\sft`
2. 打开根目录下的一个运行入口文件。
3. 点击编辑器右上角的 **Run Python File**，或者按 `F5`。
4. 在代码里打断点即可调试。

## 入口文件

- `run_compare.py`
  - 同一个问题同时跑基座模型和 LoRA 微调模型。
  - 最适合观察训练效果。

- `run_base.py`
  - 只跑微调前的 `Qwen2.5-3B-Instruct`。

- `run_lora.py`
  - 跑 `基座模型 + outputs/qwen25-3b-medical-lora`。

- `run_merged.py`
  - 跑 `outputs/qwen25-3b-medical-merged`。

## 修改问题

打开任意入口文件，修改顶部的 `QUESTION`：

```python
QUESTION = "糖尿病患者空腹血糖长期偏高，应该如何处理？"
```

保存后直接运行这个 Python 文件即可。

## 注意

第一次加载模型会比较慢。运行前建议关闭占用 GPU 的其他程序。如果 VSCode 没有自动识别解释器，请手动选择：

```text
C:\Users\CarlorYonng\Documents\sft\.conda\sft-med\python.exe
```
