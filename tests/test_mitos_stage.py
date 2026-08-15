"""
MITOS 分期（Milano-Torino Staging）单元测试

覆盖四个功能域（行走 / 上肢 / 吞咽 / 呼吸）的丧失组合，
以及阈值边界（walking ≤ 1、上肢需两项同时 ≤ 1、NIV 触发呼吸域）。
"""
from datetime import date

import pytest

from c9agent.data.patient_schema import (
    ALSPatientData, ALSFRSR_Observation, RespiratoryData,
    OnsetSite, Sex,
)
from c9agent.utils.als_calculators import calc_mitos_stage

# 各 ALSFRS-R 子项默认满分（无功能域丧失）
_FULL = dict(
    speech=4, salivation=4, swallowing=4,
    handwriting=4, cutting_food=4, dressing_hygiene=4,
    turning_in_bed=4, walking=4, climbing_stairs=4,
    dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
)


def _build(overrides: dict | None = None, *, niv_hours: float | None = None) -> ALSPatientData:
    """构造带单次 ALSFRS-R 记录的患者；niv_hours 设置无创通气使用时长。"""
    sub = dict(_FULL)
    if overrides:
        sub.update(overrides)
    record = ALSFRSR_Observation(date=date(2026, 1, 1), total_score=24, **sub)
    resp = RespiratoryData(niv_usage_hours_per_day=niv_hours) if niv_hours is not None else None
    return ALSPatientData(
        patient_id="MITOS", sex=Sex.MALE, age_at_onset=60,
        onset_site=OnsetSite.SPINAL_CERVICAL, diagnosis_date=date(2026, 1, 1),
        alsfrsr_records=[record], respiratory=resp,
    )


@pytest.mark.parametrize("overrides,niv_hours,expected", [
    ({}, None, 0),                                                                  # 0 域丧失 -> Stage 0
    ({"walking": 1}, None, 1),                                                     # 行走丧失 -> Stage 1
    ({"walking": 1, "swallowing": 1}, None, 2),                                    # + 吞咽 -> Stage 2
    ({"walking": 1, "swallowing": 1, "cutting_food": 1, "dressing_hygiene": 1}, None, 3),  # + 上肢 -> Stage 3
    ({  # 四域全丧失 -> Stage 4
        "walking": 1, "swallowing": 1,
        "cutting_food": 1, "dressing_hygiene": 1, "respiratory_insufficiency": 1,
    }, None, 4),
    ({}, 23.0, 1),                                                                  # 仅 NIV -> 呼吸域丧失 -> Stage 1
])
def test_mitos_stage_by_loss_count(overrides, niv_hours, expected):
    assert calc_mitos_stage(_build(overrides, niv_hours=niv_hours)) == expected


def test_mitos_upper_limb_requires_both_subitems():
    """上肢域丧失要求 cutting_food 与 dressing_hygiene 同时 ≤ 1。"""
    assert calc_mitos_stage(_build({"cutting_food": 1})) == 0          # 仅一项 -> 不丧失
    assert calc_mitos_stage(_build({"cutting_food": 1, "dressing_hygiene": 1})) == 1


def test_mitos_walking_threshold():
    """行走域丧失阈值为 ≤ 1，等于 2 不算丧失。"""
    assert calc_mitos_stage(_build({"walking": 2})) == 0
    assert calc_mitos_stage(_build({"walking": 1})) == 1


def test_mitos_respiratory_subitem_threshold():
    """呼吸域丧失阈值为 respiratory_insufficiency ≤ 1。"""
    assert calc_mitos_stage(_build({"respiratory_insufficiency": 2})) == 0
    assert calc_mitos_stage(_build({"respiratory_insufficiency": 1})) == 1


def test_mitos_no_alsfrsr_is_none():
    """无 ALSFRS-R 记录时无法分期，应为 None。"""
    p = ALSPatientData(
        patient_id="NONE", sex=Sex.FEMALE, age_at_onset=60,
        onset_site=OnsetSite.BULBAR, diagnosis_date=date.today(),
    )
    assert calc_mitos_stage(p) is None


@pytest.mark.parametrize("anchor,target_overrides", [
    ({"walking": 1}, {"swallowing": None}),                 # 锚定行走，变动吞咽域
    ({"swallowing": 1}, {"walking": None}),                 # 锚定吞咽，变动行走域
    ({"swallowing": 1}, {"cutting_food": None, "dressing_hygiene": None}),  # 变动上肢域（两项同时）
    ({"swallowing": 1}, {"respiratory_insufficiency": None}),            # 变动呼吸域（子项）
])
def test_mitos_domain_threshold_is_le_1(anchor, target_overrides):
    """MITOS 以子项 ≤ 1 判定功能域丧失（区别于 King's 的 ≤ 2）。

    固定一个已丧失域作锚点（loss 已为 1），再令目标域代表项
    恰好为 1（应算丧失 -> Stage 2）或 2（不算 -> Stage 1）。
    """
    at_1 = {k: 1 for k in target_overrides}
    at_2 = {k: 2 for k in target_overrides}
    p_at_1 = _build({**anchor, **at_1})
    p_at_2 = _build({**anchor, **at_2})
    assert calc_mitos_stage(p_at_1) == 2
    assert calc_mitos_stage(p_at_2) == 1
    # 关键：阈值处 1 算丧失、2 不算，二者分期必须不同
    assert calc_mitos_stage(p_at_1) != calc_mitos_stage(p_at_2)


def test_mitos_respiratory_or_no_double_count():
    """呼吸域丧失由 respiratory_insufficiency ≤ 1 或 NIV 触发，二者同时成立只算 1 个域。"""
    base = {"swallowing": 1}  # 锚定 1 个域
    # 仅子项 ≤ 1（无 NIV）
    assert calc_mitos_stage(_build({**base, "respiratory_insufficiency": 1})) == 2
    # 仅 NIV（子项 = 4，不触发子项分支）
    assert calc_mitos_stage(_build({**base, "respiratory_insufficiency": 4}, niv_hours=23.0)) == 2
    # 子项 ≤ 1 且 NIV 同时成立 -> 呼吸域仍只算 1 个（不重复计数）
    assert calc_mitos_stage(_build({**base, "respiratory_insufficiency": 1}, niv_hours=23.0)) == 2
