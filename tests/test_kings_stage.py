"""
King's 临床分期单元测试

覆盖各受累区域数、营养/呼吸衰竭（Stage 4）以及
属性 kings_stage 与 calc_kings_stage() 的一致性。
"""
from datetime import date

import pytest

from c9agent.data.patient_schema import (
    ALSPatientData, ALSFRSR_Observation, RespiratoryData,
    Milestone, MilestoneType, OnsetSite, Sex,
)
from c9agent.utils.als_calculators import calc_kings_stage

# 各 ALSFRS-R 子项默认满分（任一区域都不受累）
_FULL = dict(
    speech=4, salivation=4, swallowing=4,
    handwriting=4, cutting_food=4, dressing_hygiene=4,
    turning_in_bed=4, walking=4, climbing_stairs=4,
    dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
)


def _build(overrides: dict | None = None, *, niv_dep: bool = False,
           gastrostomy: bool = False) -> ALSPatientData:
    """构造一个带单次 ALSFRS-R 记录的患者，子项分数可覆盖。"""
    sub = dict(_FULL)
    if overrides:
        sub.update(overrides)
    record = ALSFRSR_Observation(date=date(2026, 1, 1), total_score=24, **sub)

    milestones: list[Milestone] = []
    if gastrostomy:
        milestones.append(Milestone(
            milestone_type=MilestoneType.GASTROSTOMY, date=date(2026, 1, 1)))
    if niv_dep:
        milestones.append(Milestone(
            milestone_type=MilestoneType.NIV_DEPENDENCY, date=date(2026, 1, 1)))

    return ALSPatientData(
        patient_id="KING", sex=Sex.MALE, age_at_onset=60,
        onset_site=OnsetSite.SPINAL_CERVICAL,
        diagnosis_date=date(2026, 1, 1),
        alsfrsr_records=[record],
        milestones=milestones,
    )


@pytest.mark.parametrize("overrides,expected", [
    ({}, 1),                                   # 0 区域受累 -> Stage 1
    ({"speech": 1}, 1),                        # 球部 1 区域 -> Stage 1
    ({"speech": 1, "walking": 1}, 2),          # 球部 + 下肢 -> Stage 2
    ({"speech": 1, "walking": 1, "dyspnea": 1}, 3),  # + 呼吸 -> Stage 3
    ({"speech": 1, "handwriting": 1, "walking": 1, "dyspnea": 1}, 4),  # 4 区域 -> Stage 4
])
def test_kings_stage_by_region_count(overrides, expected):
    p = _build(overrides)
    assert p.kings_stage == expected


def test_kings_stage_stage4_niv_dependency():
    """NIV 依赖（>22h/天）应直接判为 Stage 4。"""
    p = _build(niv_dep=True)
    assert p.kings_stage == 4


def test_kings_stage_stage4_gastrostomy():
    """胃造瘘（营养衰竭）应直接判为 Stage 4。"""
    p = _build(gastrostomy=True)
    assert p.kings_stage == 4


def test_kings_stage_no_alsfrsr_is_none():
    """无 ALSFRS-R 记录时无法分期，应为 None。"""
    p = ALSPatientData(
        patient_id="NONE", sex=Sex.FEMALE, age_at_onset=60,
        onset_site=OnsetSite.BULBAR, diagnosis_date=date.today(),
    )
    assert p.kings_stage is None


def test_property_consistent_with_calculator():
    """属性 kings_stage 必须与 calc_kings_stage() 始终一致（单一事实来源）。"""
    cases = [
        _build({}),
        _build({"speech": 1, "walking": 1}),
        _build({"speech": 1, "handwriting": 1, "walking": 1, "dyspnea": 1}),
        _build(niv_dep=True),
        _build(gastrostomy=True),
    ]
    for p in cases:
        assert p.kings_stage == calc_kings_stage(p)


@pytest.mark.parametrize("anchor,target_field", [
    ({"walking": 1}, "speech"),           # 锚定下肢，变动球部代表项
    ({"walking": 1}, "cutting_food"),     # 锚定下肢，变动上肢代表项
    ({"walking": 1}, "dyspnea"),          # 锚定下肢，变动呼吸代表项
    ({"swallowing": 1}, "walking"),       # 锚定球部，变动下肢
])
def test_kings_region_threshold_is_le_2(anchor, target_field):
    """King's 以子项 ≤ 2 判定区域受累（区别于 MITOS 的 ≤ 1）。

    固定一个已受累区域作锚点（Stage 已为 1），再令目标区域代表项
    恰好为 2（应算受累 -> Stage 2）或 3（不算 -> Stage 1）。
    """
    p_at_2 = _build({**anchor, target_field: 2})
    p_at_3 = _build({**anchor, target_field: 3})
    assert p_at_2.kings_stage == 2
    assert p_at_3.kings_stage == 1
    # 关键：阈值处 2 算受累、3 不算，二者分期必须不同
    assert p_at_2.kings_stage != p_at_3.kings_stage


def test_kings_upper_limb_uses_or():
    """King's 上肢域：handwriting 或 cutting_food 任一 ≤ 2 即算受累（OR，非 AND）。

    区别于 MITOS（要求两项同时 ≤ 1）。
    """
    assert _build({"cutting_food": 2}).kings_stage == 1   # 仅 cutting_food=2
    assert _build({"handwriting": 2}).kings_stage == 1    # 仅 handwriting=2
    # 两项均 > 2 -> 上肢域不受累（无其它区域 -> Stage 1）
    assert _build({"cutting_food": 3, "handwriting": 3}).kings_stage == 1


def test_kings_bulbar_uses_or():
    """King's 球部域：speech 或 swallowing 任一 ≤ 2 即算受累（OR）。"""
    assert _build({"speech": 2}).kings_stage == 1
    assert _build({"swallowing": 2}).kings_stage == 1
