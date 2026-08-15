"""
c9agent/config.py — 系统全局配置

所有模块都从这里读取配置，修改一处即可影响整个系统。
参考 newagent/config.py 的模式，但扩展为多模型 + 多数据源配置。
"""

from pathlib import Path

# ============================================================================
# LLM 配置 — 本地 Ollama 部署
# ============================================================================
LLM_CONFIG = {
    "primary": {
        "provider": "ollama",
        "api_url": "http://localhost:11434/v1/chat/completions",
        "model": "deepseek-v2:16b",        # 你已有的本地模型
        "max_tokens": 4000,
        "temperature": 0.2,                 # 临床推理需要低温度，减少随机性
        "timeout": 120.0,                   # 复杂推理可能需要较长时间
    },
    # 备选：如果需要更强的中文医疗文本能力，可以用 qwen3
    # "alternative": {
    #     "provider": "ollama",
    #     "api_url": "http://localhost:11434/v1/chat/completions",
    #     "model": "qwen3:32b",
    #     "max_tokens": 4000,
    #     "temperature": 0.2,
    #     "timeout": 120.0,
    # },
}

# ============================================================================
# 数据路径配置
# ============================================================================
DATA_CONFIG = {
    # PRO-ACT 公开数据集（3220例ALS患者，用于训练生存预测模型）
    "proact_dir": "./data/proact/",
    # 合成测试数据输出目录
    "synthetic_dir": "./data/synthetic/",
    # ALS 临床术语字典（中英对照）
    "terminology_file": "./data/als_terminology.json",
    # 文献缓存（避免重复请求 PubMed）
    "literature_cache": "./data/literature_cache.db",
}

# ============================================================================
# 模型配置
# ============================================================================
MODEL_CONFIG = {
    # ★ 模型系数文件路径 —— 训练后自动更新，也可手动编辑 ★
    "coefficients_file": str(Path(__file__).resolve().parent.parent / "data" / "model_coefficients.json"),
    # 生存预测模型参数（使用特征列 + 训练参数）
    "survival": {
        "test_size": 0.2,
        "random_state": 42,
        "features": [
            "age_at_onset", "onset_site_bulbar", "alsfrsr_slope",
            "fvc_percent", "diagnostic_delay_months", "has_c9orf72", "has_sod1",
        ],
    },
    # ALSFRS-R 轨迹预测模型
    "trajectory": {
        "model_type": "fnn",                # 前馈神经网络
        "horizon_months": [1, 3, 6, 12],   # 预测未来 1/3/6/12 个月
        "target_mae": 2.65,                 # 文献报告的基线 MAE
    },
}

# ============================================================================
# Agent 配置
# ============================================================================
AGENT_CONFIG = {
    # 自我反思循环的最大迭代次数（参考 DeepRare 的 N 参数）
    "max_reflection_iterations": 3,
    # 并行执行的 Agent 列表
    "parallel_agents": ["genotype", "prognosis", "literature"],
    # 文献检索的最大返回结果数
    "literature_max_results": 20,
    # 基因变异过滤阈值
    "gnomad_af_threshold": 0.01,           # gnomAD 频率 >1% 视为良性
}

# ============================================================================
# ALS 关键基因列表
# ============================================================================
ALS_GENES = [
    "C9orf72", "SOD1", "TARDBP", "FUS", "TBK1",
    "OPTN", "VCP", "UBQLN2", "SQSTM1", "CHCHD10",
    "PFN1", "HNRNPA1", "MATR3", "TUBA4A", "NEK1",
    "C21orf2", "KIF5A", "ANXA11", "CCNF", "FIG4",
    "ALS2", "SETX", "SPG11", "ATXN2",
]
