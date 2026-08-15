# C9Agent

ALS（肌萎缩侧索硬化，渐冻症）智能分析系统 —— 面向已知 ALS 确诊人群的预后预测、进展分析和个性化报告生成。

## 安装

### 核心功能（推荐）

```bash
pip install -e .
# 或
pip install -r requirements.txt
```

### 完整功能（含 API 服务 & 文档提取）

```bash
pip install -e ".[full]"
# 或
pip install -r requirements-full.txt
```

### 开发依赖

```bash
pip install -e ".[dev]"
```

## 快速开始

### 1. 单患者分析（合成数据测试）

```bash
python scripts/run_single_patient.py --synthetic --no-reflection --output report.json --markdown report.md
```

### 2. 队列分析

```bash
python scripts/run_cohort_analysis.py --synthetic --n 100 -o cohort_report.json
```

### 3. 训练/重置 CoxPH 模型

```bash
# 用合成数据训练
python scripts/train_models.py --synthetic --n 500

# 重置为文献默认值
python scripts/train_models.py --reset
```

### 4. 启动 API 服务（需要完整依赖）

```bash
python scripts/run_api_server.py
```

访问文档：`http://localhost:8000/docs`

## 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
c9agent/
├── agents/           # 智能体：表型提取、预后预测、文献检索、队列分析
├── core/             # 中央调度器、记忆银行、反思循环、报告生成器
├── data/             # 患者数据模型、临床术语、知识库、合成数据
├── models/           # 生存预测模型、基因型-表型关联分析
└── utils/            # LLM 客户端、临床计算器

scripts/              # 可执行脚本
 tests/               # 测试套件
```

## 注意事项

- 当前版本使用文献汇总的 CoxPH 系数和合成数据进行开发与验证。
- LLM 默认配置为本地 Ollama 服务，可在 `c9agent/config.py` 中修改。
- 本项目仅供研究与开发使用，尚未在真实临床队列上完成外部验证，**不应用于真实临床决策**。
