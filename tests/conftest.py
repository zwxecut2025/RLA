"""
pytest 配置文件 —— 为整个测试套件提供共享的 fixtures。
"""
import pytest
from c9agent.data.synthetic_data import SyntheticALSGenerator
from c9agent.data.patient_schema import ALSPatientData, ALSPatientInput


@pytest.fixture
def synthetic_generator():
    """合成数据生成器（固定种子，可复现）"""
    return SyntheticALSGenerator(seed=42)


@pytest.fixture
def sample_patient(synthetic_generator) -> ALSPatientData:
    """生成一个标准的模拟 ALS 患者"""
    return synthetic_generator.generate_patient(months_since_onset=18)


@pytest.fixture
def sample_input(sample_patient) -> ALSPatientInput:
    """包装为输入格式"""
    return ALSPatientInput(
        patient_id=sample_patient.patient_id,
        structured_data=sample_patient,
    )


@pytest.fixture
def bulbar_patient(synthetic_generator) -> ALSPatientData:
    """生成一个球部起病的模拟患者（预后更差）"""
    patient = synthetic_generator.generate_patient(months_since_onset=6)
    # 强制球部起病
    from c9agent.data.patient_schema import OnsetSite
    patient.onset_site = OnsetSite.BULBAR
    return patient
