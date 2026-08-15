"""
c9agent/data/synthetic_data.py — 合成 ALS 患者数据生成

在没有真实临床数据的情况下，生成模拟的 ALS 患者用于开发和测试。
数据分布参考真实 ALS 流行病学文献:
- 发病率: ~2/10万/年
- 男女比: 1.3:1
- 球部起病: ~25-30%
- C9orf72 突变: ~5-7% (欧美) / ~1-3% (亚洲)
- SOD1 突变: ~2% (家族性 ALS 中 ~20%)
- 中位生存期: ~30个月（从起病算）
- 诊断延迟: 中位 ~12个月
"""

import random
import uuid
from datetime import date, timedelta
from typing import Optional
from c9agent.data.patient_schema import (
    ALSPatientData, ALSPatientInput, ALSFRSR_Observation,
    GeneVariant, RespiratoryData, Medication, Milestone,
    OnsetSite, Sex, VariantType, Zygosity, ACMGClass, MilestoneType,
)


class SyntheticALSGenerator:
    """
    合成 ALS 患者数据生成器。

    使用方式:
        gen = SyntheticALSGenerator(seed=42)
        patient = gen.generate_patient(months_since_onset=12)
        gen.save_to_json(patient, "sample_patient.json")
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._used_ids: set[str] = set()

    def generate_patient(
        self,
        patient_id: str = None,
        months_since_onset: int = 12,
        include_genetics: bool = True,
    ) -> ALSPatientData:
        """
        生成一个模拟 ALS 患者。

        参数:
            patient_id: 患者编号（None = 自动生成）
            months_since_onset: 从发病到现在经过的月数
            include_genetics: 是否生成基因变异数据
        """
        if patient_id is None:
            patient_id = f"SYNTH-{uuid.uuid4().hex[:8].upper()}"
        self._used_ids.add(patient_id)

        # —— 人口学 ——
        sex = random.choices(
            [Sex.MALE, Sex.FEMALE], weights=[0.56, 0.44], k=1
        )[0]
        age = max(30, min(85, round(random.gauss(58, 12))))

        # —— 发病部位 ——
        onset = random.choices(
            [OnsetSite.SPINAL_CERVICAL, OnsetSite.SPINAL_LUMBAR,
             OnsetSite.BULBAR, OnsetSite.RESPIRATORY],
            weights=[0.35, 0.30, 0.28, 0.07], k=1
        )[0]

        # —— 时间线 ——
        today = date.today()
        symptom_date = today - timedelta(days=int(months_since_onset * 30.44))
        diag_delay = max(3, round(random.gauss(12, 8)))  # 诊断延迟(月)
        diagnosis_date = symptom_date + timedelta(days=int(diag_delay * 30.44))

        # —— ALSFRS-R 纵向记录 ——
        alsfrsr_records = self._generate_alsfrsr_trajectory(
            onset=onset,
            symptom_date=symptom_date,
            diagnosis_date=diagnosis_date,
            months_since_onset=months_since_onset,
            age=age,
        )

        # —— 呼吸 ——
        latest_fvc = max(20, min(120, round(random.gauss(100 - months_since_onset * 1.5, 15))))
        respiratory = RespiratoryData(
            fvc_percent_predicted=latest_fvc,
            fvc_date=today - timedelta(days=90),
            niv_usage_hours_per_day=(random.choice([0, 0, 0, 0, 4, 8, 12, 22])
                                     if months_since_onset > 24 else None),
            invasive_ventilation=random.random() < 0.05 and months_since_onset > 36,
        )

        # —— 遗传 ——
        genetic_variants = []
        if include_genetics and random.random() < 0.15:  # 15%概率有ALS基因变异
            gene = random.choices(
                ["C9orf72", "SOD1", "TARDBP", "FUS", "TBK1"],
                weights=[0.45, 0.25, 0.15, 0.10, 0.05], k=1
            )[0]
            variant = self._generate_variant(gene)
            genetic_variants.append(variant)

        # —— 用药 ——
        medications = [
            Medication(
                drug_name="Riluzole",
                start_date=diagnosis_date + timedelta(days=30),
                dosage="50mg bid",
            )
        ]
        if random.random() < 0.3:
            medications.append(
                Medication(
                    drug_name="Edaravone",
                    start_date=diagnosis_date + timedelta(days=90),
                    dosage="60mg iv, 14d on / 14d off",
                )
            )

        # —— 里程碑 ——
        milestones = [Milestone(milestone_type=MilestoneType.DIAGNOSIS, date=diagnosis_date)]
        if months_since_onset > 24 and random.random() < 0.3:
            wheel_date = symptom_date + timedelta(days=int(24 * 30.44))
            milestones.append(Milestone(milestone_type=MilestoneType.LOSS_OF_AMBULATION, date=wheel_date))

        return ALSPatientData(
            patient_id=patient_id,
            sex=sex,
            age_at_onset=age,
            onset_site=onset,
            diagnosis_date=diagnosis_date,
            symptom_onset_date=symptom_date,
            alsfrsr_records=alsfrsr_records,
            respiratory=respiratory,
            genetic_variants=genetic_variants,
            family_history_als=random.random() < 0.1,
            medications=medications,
            milestones=milestones,
            clinical_notes=None,
        )

    def _generate_alsfrsr_trajectory(
        self, onset, symptom_date, diagnosis_date, months_since_onset, age
    ) -> list[ALSFRSR_Observation]:
        """
        生成模拟的 ALSFRS-R 纵向轨迹。

        ALS 进展近似线性下降，但有患者间差异。
        典型下降速率:
        - 球部起病: ~1.2 分/月
        - 脊髓起病: ~0.8 分/月
        - 快速进展型: >1.5 分/月
        """
        if onset == OnsetSite.BULBAR:
            base_slope = random.gauss(1.2, 0.4)
        else:
            base_slope = random.gauss(0.8, 0.3)

        # 加一些随机变异性
        slope = max(0.2, base_slope)
        records = []

        # 每3个月一条记录
        num_records = max(2, months_since_onset // 3)
        for i in range(num_records + 1):
            record_date = symptom_date + timedelta(days=int(i * 90))
            if record_date > date.today():
                break

            months = i * 3
            total = max(0, 48 - (months * slope))
            total += random.randint(-3, 3)  # 测量噪声
            total = max(0, min(48, int(total)))

            # 根据起病部位，某些子项下降更快
            if onset == OnsetSite.BULBAR:
                speech = max(0, 4 - int(months * slope * 0.4))
                swallowing = max(0, 4 - int(months * slope * 0.4))
                walking = max(0, 4 - int(months * slope * 0.2))
            else:
                speech = max(0, 4 - int(months * slope * 0.15))
                swallowing = max(0, 4 - int(months * slope * 0.15))
                walking = max(0, 4 - int(months * slope * 0.35))

            records.append(ALSFRSR_Observation(
                date=record_date,
                total_score=total,
                speech=speech, salivation=max(0, 4 - int(months * slope * 0.3)),
                swallowing=swallowing,
                handwriting=max(0, 4 - int(months * slope * 0.35)),
                cutting_food=max(0, 4 - int(months * slope * 0.35)),
                dressing_hygiene=max(0, 4 - int(months * slope * 0.35)),
                turning_in_bed=max(0, 4 - int(months * slope * 0.3)),
                walking=walking,
                climbing_stairs=max(0, 4 - int(months * slope * 0.35)),
                dyspnea=max(0, 4 - int(months * slope * 0.2)),
                orthopnea=max(0, 4 - int(months * slope * 0.2)),
                respiratory_insufficiency=max(0, 4 - int(months * slope * 0.2)),
            ))

        return records

    def _generate_variant(self, gene: str) -> GeneVariant:
        """生成一个模拟的基因变异"""
        if gene == "C9orf72":
            return GeneVariant(
                gene="C9orf72",
                variant_type=VariantType.REPEAT_EXPANSION,
                hgvs_c="c.-45+162GGGGCC_repeat_expansion",
                zygosity=Zygosity.HETEROZYGOUS,
                acmg_classification=ACMGClass.PATHOGENIC,
                gnomad_af=0.001,
                description="GGGGCC hexanucleotide repeat expansion (HRE) >30 repeats",
            )
        elif gene == "SOD1":
            return GeneVariant(
                gene="SOD1",
                variant_type=VariantType.MISSENSE,
                hgvs_c="NM_000454.5:c.14C>T",
                hgvs_p="NP_000445.1:p.Ala5Val",
                zygosity=Zygosity.HETEROZYGOUS,
                acmg_classification=ACMGClass.PATHOGENIC,
                gnomad_af=0.0001,
            )
        else:
            return GeneVariant(
                gene=gene,
                variant_type=random.choice([VariantType.MISSENSE, VariantType.NONSENSE]),
                hgvs_c=f"{gene}:c.{random.randint(1,1000)}G>A",
                zygosity=Zygosity.HETEROZYGOUS,
                acmg_classification=ACMGClass.VUS,
                gnomad_af=random.uniform(0.0001, 0.01),
            )

    def generate_cohort(self, n: int = 100) -> list[ALSPatientData]:
        """生成一个模拟患者队列"""
        return [self.generate_patient(
            months_since_onset=random.randint(1, 60)
        ) for _ in range(n)]
