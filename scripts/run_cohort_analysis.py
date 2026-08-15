#!/usr/bin/env python
"""
run_cohort_analysis.py — ALS 队列分析入口

用法:
    # 用合成队列测试
    python scripts/run_cohort_analysis.py --synthetic --n 100

    # 用 JSON 导入真实队列
    python scripts/run_cohort_analysis.py --input cohort.json

    # 输出
    python scripts/run_cohort_analysis.py --synthetic -o cohort_report.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c9agent.data.synthetic_data import SyntheticALSGenerator
from c9agent.agents.cohort_agent import CohortAgent
from c9agent.models.survival_models import CoxPHSurvival


def main():
    parser = argparse.ArgumentParser(
        description="C9Agent — ALS 队列分析"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true",
                     help="用合成数据")
    src.add_argument("--input", type=str,
                     help="JSON 队列文件路径")

    parser.add_argument("--n", type=int, default=100,
                        help="合成队列大小")
    parser.add_argument("--output", "-o", type=str, default="cohort_report.json")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # —— 准备队列 ——
    if args.synthetic:
        gen = SyntheticALSGenerator(seed=args.seed)
        cohort = gen.generate_cohort(n=args.n)
        print(f"合成队列: {len(cohort)} 例 ALS 患者")

        # 合成生存数据
        model = CoxPHSurvival()
        survival_data = {}
        events = {}
        for p in cohort:
            pred = model.predict_from_patient(p)
            survival_data[p.patient_id] = pred.median_survival_months
            events[p.patient_id] = True  # 简化：所有患者观察到死亡
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = json.load(f)
        from c9agent.data.patient_schema import ALSPatientData
        cohort = [ALSPatientData(**p) for p in raw["patients"]]
        survival_data = raw.get("survival_data", {})
        events = raw.get("events", {})

    # —— 运行分析 ——
    print("分析中...")
    agent = CohortAgent()
    result = agent.execute(
        cohort=cohort,
        survival_data=survival_data,
        events=events,
    )

    # —— 输出 ——
    print(f"\n=== 队列概览 (n={result.data['cohort_size']}) ===")
    desc = result.data["description"]
    print(f"平均发病年龄: {desc['mean_age']}±{desc['std_age']}岁")
    print(f"范围: {desc['min_age']}-{desc['max_age']}岁")
    print(f"球部起病: {desc['bulbar_pct']:.1f}%")
    print(f"平均ΔFS: {desc['mean_slope']}分/月")
    print(f"快进展: {desc['fast_progression_pct']:.1f}%")
    print(f"慢进展: {desc['slow_progression_pct']:.1f}%")

    if desc.get("gene_counts"):
        print(f"\n基因变异检出:")
        for gene, cnt in sorted(desc["gene_counts"].items(),
                                 key=lambda x: -x[1]):
            print(f"  {gene}: {cnt}例 ({cnt/len(cohort)*100:.1f}%)")

    # 基因型-表型关联
    assocs = result.data.get("gene_phenotype_associations", [])
    if assocs:
        print(f"\n=== 基因-表型关联 ({len(assocs)} 个显著关联) ===")
        for a in assocs[:5]:
            sig = "***" if a["p_value"] < 0.001 else "**" if a["p_value"] < 0.01 else "*"
            print(f"  {a['gene']} × {a['phenotype']}: OR={a['odds_ratio']:.2f}, "
                  f"p={a['p_value']:.4f} {sig}")

    # 分层生存
    stratified = result.data.get("stratified_survival", {})
    if stratified:
        print(f"\n=== 分层生存 ===")
        for name, curve in stratified.items():
            if curve:
                print(f"  {name}: 中位生存={curve['median_survival']}月 "
                      f"(n={curve['n_total']}, events={curve['n_events']})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result.data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()
