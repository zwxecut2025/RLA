"""
测试患者数据模型 —— 验证 Pydantic 校验和自动计算字段
"""
from c9agent.data.patient_schema import (
    ALSPatientData, ALSFRSR_Observation, GeneVariant,
    OnsetSite, Sex, VariantType, Zygosity, ACMGClass,
)
from c9agent.utils.als_calculators import (
    calc_alsfrsr_slope, classify_progression,
    calc_kings_stage, calc_mitos_stage, estimate_median_survival,
)


class TestALSPatientData:
    """测试患者数据模型的基本功能"""

    def test_create_minimal_patient(self):
        """测试创建最简患者"""
        from datetime import date
        patient = ALSPatientData(
            patient_id="TEST001",
            sex=Sex.MALE,
            age_at_onset=55,
            onset_site=OnsetSite.SPINAL_CERVICAL,
            diagnosis_date=date.today(),
        )
        assert patient.patient_id == "TEST001"
        assert patient.alsfrsr_slope is None  # 无记录，无法计算
        assert patient.kings_stage is None

    def test_auto_computed_fields(self, sample_patient):
        """测试自动计算的临床参数"""
        p = sample_patient
        assert p.latest_alsfrsr is not None
        assert 0 <= p.latest_alsfrsr.total_score <= 48
        if p.alsfrsr_slope is not None:
            assert p.alsfrsr_slope > 0  # ALS 只能下降
        assert p.progression_category in ("fast", "moderate", "slow", None)

    def test_bulbar_onset_computed(self, bulbar_patient):
        """球部起病患者 is_bulbar_onset 应为 True"""
        assert bulbar_patient.is_bulbar_onset is True

    def test_no_alsfrsr_patient(self):
        """无 ALSFRS-R 数据的患者"""
        from datetime import date
        patient = ALSPatientData(
            patient_id="NO_DATA",
            sex=Sex.FEMALE,
            age_at_onset=60,
            onset_site=OnsetSite.BULBAR,
            diagnosis_date=date.today(),
        )
        assert patient.latest_alsfrsr is None
        assert patient.alsfrsr_slope is None
        assert patient.kings_stage is None


class TestALSCalculators:
    """测试 ALS 临床计算器"""

    def test_progression_classification(self):
        assert classify_progression(1.5) == "fast"
        assert classify_progression(0.6) == "moderate"
        assert classify_progression(0.3) == "slow"

    def test_estimate_survival_bulbar(self):
        """球部起病 + 高龄 = 预后差"""
        result = estimate_median_survival(
            onset_site="bulbar", age_at_onset=65,
            alsfrsr_slope=1.2, fvc_percent=60, has_pathogenic_variant=True,
        )
        assert result["risk_level"] == "high"
        # 这么多不良因子，中位生存应 < 18个月
        assert result["median_months"] < 18

    def test_estimate_survival_favorable(self):
        """年轻的脊髓起病患者预后更好"""
        result = estimate_median_survival(
            onset_site="spinal_cervical", age_at_onset=40,
            alsfrsr_slope=0.3, fvc_percent=95, has_pathogenic_variant=False,
        )
        assert result["risk_level"] in ("low", "moderate")
        assert result["median_months"] > 24


class TestSyntheticData:
    """测试合成数据生成器"""

    def test_generate_single(self, synthetic_generator):
        patient = synthetic_generator.generate_patient(months_since_onset=12)
        assert patient.patient_id.startswith("SYNTH-")
        assert 30 <= patient.age_at_onset <= 85
        assert len(patient.alsfrsr_records) >= 2

    def test_generate_cohort(self, synthetic_generator):
        cohort = synthetic_generator.generate_cohort(n=10)
        assert len(cohort) == 10
        # 所有患者 ID 唯一
        ids = [p.patient_id for p in cohort]
        assert len(set(ids)) == 10
