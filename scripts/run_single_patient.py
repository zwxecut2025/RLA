#!/usr/bin/env python
"""
run_single_patient.py — ALS 智能体单患者分析入口

用法:
    # 用合成数据测试
    python scripts/run_single_patient.py --synthetic

    # 用 JSON 文件输入
    python scripts/run_single_patient.py --input patient.json

    # 用自由文本输入
    python scripts/run_single_patient.py --text "64岁男性，球部起病ALS，ALSFRS-R=34..."

    # 输出到文件
    python scripts/run_single_patient.py --synthetic --output report.json

这是 Phase 1 的主入口 —— 实现了 DeepRare 论文中的完整流程:
输入 → PhenotypeAgent → PrognosisAgent → LLM证据合成 → 报告输出
"""

import sys
import json
import argparse
from pathlib import Path

# 把项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from c9agent.core.orchestrator import CentralOrchestrator
from c9agent.data.patient_schema import ALSPatientInput, ALSPatientData
from c9agent.data.synthetic_data import SyntheticALSGenerator


def main():
    parser = argparse.ArgumentParser(
        description="C9Agent — ALS 渐冻症智能分析系统",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--synthetic", action="store_true",
                             help="使用模拟 ALS 患者数据")
    input_group.add_argument("--input", type=str,
                             help="JSON 输入文件路径")
    input_group.add_argument("--text", type=str,
                             help="自由文本临床描述")

    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出报告文件路径 (JSON)")
    parser.add_argument("--markdown", "-m", type=str, default=None,
                        help="输出 Markdown 报告文件路径")
    parser.add_argument("--no-reflection", action="store_true",
                        help="禁用自我反思循环")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="安静模式，不打印进度")
    parser.add_argument("--seed", type=int, default=42,
                        help="合成数据的随机种子")

    args = parser.parse_args()

    # —— 准备输入 ——
    if args.synthetic:
        gen = SyntheticALSGenerator(seed=args.seed)
        patient = gen.generate_patient(months_since_onset=18)
        # 保存合成数据供参考
        synth_path = PROJECT_ROOT / "data" / "synthetic"
        synth_path.mkdir(parents=True, exist_ok=True)
        with open(synth_path / "latest_synthetic.json", "w", encoding="utf-8") as f:
            json.dump(patient.model_dump(mode="json"), f, ensure_ascii=False,
                      indent=2, default=str)
        if not args.quiet:
            print(f"[合成患者] {patient.patient_id}: "
                  f"{patient.sex.value}, {patient.age_at_onset}岁, "
                  f"{patient.onset_site.value}起病, "
                  f"ALSFRS-R={patient.latest_alsfrsr.total_score if patient.latest_alsfrsr else 'N/A'}")
        input_data = ALSPatientInput(
            patient_id=patient.patient_id, structured_data=patient
        )

    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 尝试解析为 ALSPatientData
        if "patient_id" in raw:
            input_data = ALSPatientInput(
                patient_id=raw["patient_id"],
                structured_data=ALSPatientData(**raw) if "alsfrsr_records" in raw else None,
                free_text=raw.get("clinical_notes"),
            )
        else:
            input_data = ALSPatientInput(
                patient_id="IMPORTED",
                free_text=json.dumps(raw, ensure_ascii=False),
            )

    elif args.text:
        input_data = ALSPatientInput(patient_id="TEXT-001", free_text=args.text)

    else:
        parser.error("必须指定 --synthetic, --input, 或 --text")

    # —— 运行分析 ——
    orchestrator = CentralOrchestrator(
        enable_reflection=not args.no_reflection
    )
    report = orchestrator.analyze_single_patient(input_data, verbose=not args.quiet)

    # —— 输出报告 ——
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nJSON 报告已保存至: {args.output}")

    if args.markdown:
        md_text = orchestrator.report_builder.to_markdown(report)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Markdown 报告已保存至: {args.markdown}")

    if not args.output and not args.markdown:
        print("\n" + "=" * 60)
        print("分析报告 (Markdown)")
        print("=" * 60)
        print(orchestrator.report_builder.to_markdown(report))


if __name__ == "__main__":
    main()
