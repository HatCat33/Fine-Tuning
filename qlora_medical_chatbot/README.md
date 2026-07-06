# QLoRA 微调智能客服机器人

基于 **Qwen-7B-Chat** 的 QLoRA（4-bit 量化 + LoRA）微调项目，使用中文医疗对话数据训练一个「智能医生客服机器人」。

## 项目概述

| 项目 | 说明 |
|------|------|
| 基座模型 | Qwen-7B-Chat（通义千问 7B 对话模型） |
| 微调方法 | QLoRA（4-bit NormalFloat4 量化 + LoRA 低秩适配） |
| 训练数据 | 中文医疗对话数据集（儿科，84,344 条）+ 自我认知数据（80 条） |
| 机器人代号 | 智能医生客服机器人小D |
| 训练平台 | AutoDL 云 GPU / 本地 GPU |
|  GPU | V100-SXM2-32GB |

## 项目结构

```
├── chatbot_qlora_ft.ipynb    # 完整训练 Notebook（数据加载 → 量化 → LoRA → 训练 → 推理）
├── self_cognition.json       # 机器人自我认知数据（80 条 QA）
├── requirements.txt          # Python 依赖
├── .gitignore
└── README.md
```

> 训练数据集 `儿科5-14000.csv` 因体积较大未包含在仓库中，请从 [中文医疗对话数据集](https://github.com/Toyhom/Chinese-medical-dialogue-data) 下载。

## 快速开始

### 1. 环境准备
```bash
# 选择镜像
[funhpc](https://www.funhpc.com)

选择：V100-SXM2-32GB 
版本：pytorch -> 2.1.0 -> 3.10 -> 12.1

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载基座模型

```bash
# 方式 A：ModelScope 下载（国内推荐）
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen-7B-Chat', cache_dir='./llm')"

# 方式 B：HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download qwen/Qwen-7B-Chat --local-dir ./llm/Qwen-7B-Chat
```

### 3. 准备数据

将医疗对话 CSV 文件放到 Notebook 同目录下，修改 Notebook 中数据加载路径。

### 4. 运行训练

```bash
打开 chatbot_qlora_ft.ipynb，按顺序执行全部 Cell
```

## 技术方案

### QLoRA 量化配置

```python
BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",     # NormalFloat4 量化类型
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, # 双重量化
)
```

### LoRA 配置

| 参数 | 值 |
|------|-----|
| 秩 (r) | 32 |
| 缩放因子 (alpha) | 16 |
| 目标模块 | c_attn, c_proj, w1, w2 |
| Dropout | 0.05 |

### 训练参数

| 参数 | 值 |
|------|-----|
| 优化器 | paged_adamw_8bit |
| 学习率 | 2e-4 |
| Batch Size | 1 |
| 最大步数 | 1000 |
| 序列长度 | 1024 |
| 梯度检查点 | 开启 |


## 训练流程

1. **环境搭建** — 安装 PyTorch + Transformers + PEFT + BitsAndBytes
2. **数据加载** — 解析 CSV 医疗对话 + JSON 自我认知数据
3. **格式转换** — 转为千问官方微调格式 `{"conversations": [...]}`
4. **量化加载** — 4-bit 加载基座模型（仅占 ~6GB 显存）
5. **LoRA 注入** — 在注意力层和 FFN 层插入低秩适配器
6. **数据预处理** — Tokenize + Padding + 构造 SupervisedDataset
7. **训练** — paged_adamw_8bit 优化，梯度检查点降低显存
8. **合并推理** — 加载 checkpoint 进行效果验证

## 微调效果

微调前（基座模型）：
- 「你好」→ 通用问候
- 「孩子积食」→ 通用建议

微调后（机器人小言）：
- 「你好」→ "您好，我是机器人小言，很高兴认识您！"
- 「孩子积食」→ 详细的儿科医疗建议，包含具体的护理步骤和温度参数

## 已知问题

- QLoRA 训练速度比普通 LoRA 慢约 30%
- 4-bit 量化可能引入轻微精度损失
- 数据集仅使用儿科子集，覆盖科室有限
- 建议进一步增加训练步数、扩充多科室数据以获得更好的泛化能力

## 参考

- [QwenLM/Qwen](https://github.com/QwenLM/Qwen) — 通义千问官方仓库
- [huggingface/peft](https://github.com/huggingface/peft) — PEFT 参数高效微调
- [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 量化库
- [Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data) — 中文医疗对话数据集
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) —— QLoRA 论文

## 其他数据集
- https://github.com/CLUEbenchmark/CLUEDatasetSearch
- https://github.com/brightmart/nlp_chinese_corpus
- https://github.com/GanjinZero/awesome_Chinese_medical_NLP
- https://github.com/smoothnlp/FinancialDatasets
- https://github.com/juand-r/entity-recognition-datasets
- https://github.com/SophonPlus/ChineseNlpCorpus
