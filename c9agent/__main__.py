"""
c9agent/__main__.py — 命令行入口

用法:
    python -m c9agent                      # 运行内置样例患者
    python -m c9agent path/to/patient.json # 分析指定患者（结构化 JSON）

JSON 需符合 c9agent.data.patient_schema.ALSPatientData 结构。
生存预测/文献/反思均依赖本地 Ollama（config.py 中配置）。
"""

import json
import sys
from datetime import date
from pathlib import Path

from c9agent.data.patient_schema import (
    ALSPatientData, ALSPatientInput,
    ALSFRSR_Observation, RespiratoryData, GeneVariant,
    OnsetSite, Sex, VariantType, Zygosity, ACMGClass,
)
from c9agent.core.orchestrator import CentralOrchestrator
from c9agent.core.report_builder import TraceableReportBuilder


def demo_patient() -> ALSPatientData:
    """内置样例患者：58岁男性、球部起病、快速进展、携带 C9orf72 致病变异。"""
    return ALSPatientData(
        patient_id="P-DEMO-001",
        sex=Sex.MALE,
        age_at_onset=58,
        onset_site=OnsetSite.BULBAR,
        diagnosis_date=date(2026, 5, 1),
        symptom_onset_date=date(2025, 11, 1),
        alsfrsr_records=[
            ALSFRSR_Observation(
                date=date(2026, 1, 1), total_score=44,
                speech=3, salivation=4, swallowing=3,
                handwriting=4, cutting_food=4, dressing_hygiene=4,
                turning_in_bed=4, walking=4, climbing_stairs=4,
                dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
            ),
            ALSFRSR_Observation(
                date=date(2026, 5, 1), total_score=32,
                speech=1, salivation=2, swallowing=1,
                handwriting=3, cutting_food=2, dressing_hygiene=3,
                turning_in_bed=4, walking=3, climbing_stairs=3,
                dyspnea=3, orthopnea=3, respiratory_insufficiency=3,
            ),
        ],
        respiratory=RespiratoryData(
            fvc_percent_predicted=62.0,
            niv_usage_hours_per_day=0.0,
            invasive_ventilation=False,
        ),
        genetic_variants=[
            GeneVariant(
                gene="C9orf72",
                variant_type=VariantType.REPEAT_EXPANSION,
                zygosity=Zygosity.HETEROZYGOUS,
                acmg_classification=ACMGClass.PATHOGENIC,
            )
        ],
        family_history_als=True,
    )


def run(patient: ALSPatientData) -> str:
    """对一个患者执行完整分析并返回 Markdown 报告。"""
    inp = ALSPatientInput(patient_id=patient.patient_id, structured_data=patient)
    orch = CentralOrchestrator(enable_reflection=True)
    report = orch.analyze_single_patient(inp, verbose=True)
    return TraceableReportBuilder().to_markdown(report)


def load_patient_json(path: Path) -> ALSPatientData:
    """从结构化 JSON 加载患者（兼容 data/ALS_patient.json 格式）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ALSPatientData(**data)


def main() -> None:
    if len(sys.argv) > 1:
        patient = load_patient_json(Path(sys.argv[1]))
    else:
        patient = demo_patient()
    print(run(patient))


if __name__ == "__main__":
    main()
