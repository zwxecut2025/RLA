#!/usr/bin/env python
"""
train_models.py — ALS 生存预测模型训练脚本

功能:
1. 用合成数据 / PRO-ACT 数据拟合 CoxPH 模型
2. 输出系数到 data/model_coefficients.json
3. 支持手动编辑系数后重新评估

用法:
    # 用合成数据训练（不需要任何外部数据）
    python scripts/train_models.py --synthetic --n 500

    # 用 PRO-ACT 数据训练（需要先下载数据）
    python scripts/train_models.py --proact ./data/proact/

    # 评估当前系数（不重新训练，只计算 C-index）
    python scripts/train_models.py --evaluate

    # 用自定义 CSV 训练
    python scripts/train_models.py --csv my_patients.csv

    # 重置为文献默认值
    python scripts/train_models.py --reset
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import date

from c9agent.data.synthetic_data import SyntheticALSGenerator
from c9agent.models.survival_models import (
    CoxPHTrainer, CoxPHSurvival, DEFAULT_COEFFICIENTS,
)
from c9agent.config import MODEL_CONFIG


# ============================================================================
# 合成训练数据生成
# ============================================================================

def generate_synthetic_training_data(n_patients: int = 500,
                                     seed: int = 42) -> pd.DataFrame:
    """
    生成用于训练 CoxPH 的合成 ALS 队列数据。

    每行 = 一个患者，包含特征列 + 生存时间和事件指示。

    返回的 DataFrame 列:
        age_per_10yr, bulbar_onset, alsfrsr_slope_per_point,
        fvc_per_10pct, diagnostic_delay_per_month,
        c9orf72_positive, sod1_positive, male,
        survival_months, event_died
    """
    print(f"生成 {n_patients} 例合成 ALS 患者...")
    gen = SyntheticALSGenerator(seed=seed)

    rows = []
    for i in range(n_patients):
        # 随机病程长度
        months = int(np.random.choice([
            np.random.randint(3, 12),   # 早期 (30%)
            np.random.randint(12, 36),  # 中期 (40%)
            np.random.randint(36, 72),  # 晚期 (30%)
        ], p=[0.3, 0.4, 0.3]))

        patient = gen.generate_patient(months_since_onset=months)

        # 特征提取（与 CoxPHSurvival._extract_features 一致）
        age_per_10yr = max(0, patient.age_at_onset - 60) / 10.0
        bulbar = 1.0 if patient.is_bulbar_onset else 0.0
        slope = patient.alsfrsr_slope or np.random.uniform(0.2, 1.5)
        fvc = (patient.respiratory.fvc_percent_predicted
               if patient.respiratory else np.random.uniform(40, 100))
        fvc_per_10pct = (fvc - 100) / 10.0
        delay = patient.diagnostic_delay_months or np.random.uniform(3, 36)
        has_c9 = 1.0 if any(
            v.gene == "C9orf72" for v in patient.genetic_variants
        ) else 0.0
        has_sod1 = 1.0 if any(
            v.gene == "SOD1" for v in patient.genetic_variants
        ) else 0.0
        male = 1.0 if patient.sex.value == "male" else 0.0

        # 生存标签: 基于特征模拟
        log_hr = (
            0.35 * age_per_10yr + 0.42 * bulbar + 0.67 * slope +
            (-0.095) * fvc_per_10pct + (-0.015) * delay +
            0.47 * has_c9 + 0.22 * has_sod1 + 0.15 * male
        )
        hr = np.exp(log_hr)
        # 基线中位生存 30 月，除 HR 得到预测中位
        median = max(3, 30 / hr)
        # 加随机噪声
        true_survival = max(1, np.random.normal(median, median * 0.3))

        # 是否已观察到死亡（模拟删失：病程长的更可能已死亡）
        is_dead = months >= true_survival

        rows.append({
            "patient_id": patient.patient_id,
            "age_per_10yr": round(age_per_10yr, 2),
            "bulbar_onset": int(bulbar),
            "alsfrsr_slope_per_point": round(slope, 2),
            "fvc_per_10pct": round(fvc_per_10pct, 1),
            "diagnostic_delay_per_month": round(delay, 1),
            "c9orf72_positive": int(has_c9),
            "sod1_positive": int(has_sod1),
            "male": int(male),
            "survival_months": round(true_survival, 1),
            "event_died": 1 if is_dead else 0,
        })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} 行, {df['event_died'].sum()} 死亡事件 "
          f"({df['event_died'].mean()*100:.0f}% 删失率)")
    return df


# ============================================================================
# 训练
# ============================================================================

def train(df: pd.DataFrame, output_file: str, **meta):
    """用 DataFrame 训练 CoxPH 并保存系数"""
    feature_cols = [
        "age_per_10yr", "bulbar_onset", "alsfrsr_slope_per_point",
        "fvc_per_10pct", "diagnostic_delay_per_month",
        "c9orf72_positive", "sod1_positive", "male",
    ]

    print(f"\n训练 CoxPH 模型 (lifelines)...")
    print(f"  样本: {len(df)}")
    print(f"  特征: {feature_cols}")

    trainer = CoxPHTrainer()
    trainer.fit(df, feature_cols=feature_cols)

    print(f"\n拟合结果:")
    for feat, coef in trainer.get_coefficients_dict()["coefficients"].items():
        hr = np.exp(coef)
        print(f"  {feat}: beta={coef:.4f}  HR={hr:.2f}")

    trainer.save(output_file, **meta)

    return trainer


# ============================================================================
# 评估
# ============================================================================

def evaluate(coefficients_file: str, df: pd.DataFrame = None):
    """评估当前系数的 C-index"""
    model = CoxPHSurvival.load(coefficients_file)

    if df is None:
        df = generate_synthetic_training_data(n_patients=200, seed=999)

    # 计算预测风险分数 vs 真实生存时间的 concordance
    feature_cols = [
        "age_per_10yr", "bulbar_onset", "alsfrsr_slope_per_point",
        "fvc_per_10pct", "diagnostic_delay_per_month",
        "c9orf72_positive", "sod1_positive", "male",
    ]

    risk_scores = []
    for _, row in df.iterrows():
        features = np.array([row[c] for c in feature_cols])
        log_hr = 0
        for i, name in enumerate(model.feature_names):
            log_hr += model.coefficients.get(name, 0) * features[i]
        risk_scores.append(log_hr)

    # Concordance: 风险高的应该生存时间短
    from scipy.stats import kendalltau
    tau, pval = kendalltau(risk_scores, -df["survival_months"].values)
    c_index = (tau + 1) / 2  # Kendall tau → C-index 近似

    print(f"\n当前系数: {Path(coefficients_file).name}")
    print(f"  模型版本: {model.version}")
    print(f"  近似 C-index: {c_index:.3f}")
    print(f"  特征: {model.feature_names}")
    print(f"  系数: {json.dumps(model.coefficients, indent=2)}")

    meta = model.metadata
    if meta.get("c_index"):
        print(f"  训练时 C-index: {meta['c_index']}")
    print(f"  来源: {meta.get('trained_on', 'unknown')}")

    return c_index


# ============================================================================
# 重置
# ============================================================================

def reset(filepath: str):
    """重置为文献默认值"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_COEFFICIENTS, f, ensure_ascii=False, indent=2)
    print(f"已重置为文献默认值: {filepath}")


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="C9Agent 模型训练 — 拟合/评估/重置 CoxPH 系数"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true",
                     help="用合成数据训练")
    src.add_argument("--csv", type=str,
                     help="用自定义 CSV 文件训练")
    src.add_argument("--proact", type=str,
                     help="用 PRO-ACT 数据训练（指定数据目录）")
    src.add_argument("--evaluate", action="store_true",
                     help="评估当前系数（不重新训练）")
    src.add_argument("--reset", action="store_true",
                     help="重置系数为文献默认值")

    parser.add_argument("--n", type=int, default=500,
                        help="合成数据的样本量 (default: 500)")
    parser.add_argument("--output", "-o", type=str,
                        default=MODEL_CONFIG.get("coefficients_file",
                                                  "data/model_coefficients.json"),
                        help="系数输出路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--version", type=str, default=None,
                        help="模型版本号")

    args = parser.parse_args()

    # —— 处理 ——

    if args.reset:
        reset(args.output)
        return

    if args.evaluate:
        evaluate(args.output)
        return

    # 训练
    version = args.version or date.today().strftime("%Y%m%d")

    if args.synthetic:
        df = generate_synthetic_training_data(n_patients=args.n, seed=args.seed)
        train(df, args.output,
              version=version,
              trained_on=f"synthetic (n={args.n})")

    elif args.csv:
        df = pd.read_csv(args.csv)
        train(df, args.output,
              version=version,
              trained_on=f"CSV: {Path(args.csv).name}")

    elif args.proact:
        proact_path = Path(args.proact)
        if not proact_path.exists():
            print(f"[ERROR] PRO-ACT 数据目录不存在: {args.proact}")
            print("请先从 https://nctu.partners.org/ProACT 下载数据")
            sys.exit(1)
        # TODO: PRO-ACT 加载器（Phase 2 完整实现）
        print("[INFO] PRO-ACT 加载器将在 Phase 2 实现，暂时用合成数据替代")
        df = generate_synthetic_training_data(n_patients=args.n, seed=args.seed)
        train(df, args.output,
              version=version,
              trained_on=f"synthetic-fallback (n={args.n})")

    # —— 快速验证 ——
    print("\n" + "=" * 60)
    print("验证: 用新系数运行一次预测")
    model = CoxPHSurvival.load(args.output)
    gen = SyntheticALSGenerator(seed=123)
    test_patient = gen.generate_patient(months_since_onset=18)
    pred = model.predict_from_patient(test_patient)
    print(f"  测试患者: {test_patient.patient_id}")
    print(f"  中位生存: {pred.median_survival_months} 月")
    print(f"  风险等级: {pred.risk_level}")
    print(f"  12月生存率: {pred.survival_prob.get(12, 'N/A')}")

    print(f"\n系数文件: {Path(args.output).resolve()}")
    print("你可以直接编辑这个 JSON 文件来修改系数，然后重新运行推理。")


if __name__ == "__main__":
    main()
