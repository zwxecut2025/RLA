"""
c9agent/utils/als_calculators.py — ALS 临床计算工具

提供 ALS 领域专用的计算函数：
- ALSFRS-R 下降速率 (ΔFS)
- King's 临床分期
- MITOS 分期（米兰都灵分期）
- 预计生存期估算（基于文献公式）
"""

from c9agent.data.patient_schema import ALSPatientData, ALSFRSR_Observation


def calc_alsfrsr_slope(records: list[ALSFRSR_Observation]) -> float | None:
    """
    计算 ALSFRS-R 下降速率。

    公式: (48 - 最新分数) / 观察时间跨度(月)
    返回: 分/月

    文献参考:
    - 正常进展: < 0.45 分/月
    - 中等进展: 0.45-0.89 分/月
    - 快速进展: > 0.89 分/月
    (Kimura et al., 2006; Labra et al., 2016)
    """
    if len(records) < 2:
        return None

    sorted_records = sorted(records, key=lambda r: r.date)
    first = sorted_records[0]
    last = sorted_records[-1]

    days = (last.date - first.date).days
    if days <= 0:
        return None

    months = days / 30.44
    delta = first.total_score - last.total_score  # 正值表示下降
    return delta / months


def classify_progression(slope: float) -> str:
    """根据 ALSFRS-R 下降速率分类进展速度"""
    if slope > 0.89:
        return "fast"
    elif slope > 0.45:
        return "moderate"
    else:
        return "slow"


def calc_kings_stage(patient: ALSPatientData) -> int | None:
    """
    King's 临床分期系统 (Roche et al., 2012)

    Stage 1 = 症状 onset（一个区域）
    Stage 2 = 第二个区域受累（A: 诊断时 B: 随访中）
    Stage 3 = 第三个区域受累
    Stage 4A = 需要胃造瘘（营养衰竭）
    Stage 4B = 需要 NIV（呼吸衰竭）

    这里做简化: 1-4 的整数
    """
    latest = patient.latest_alsfrsr
    if latest is None:
        return None

    # 统计 ALSFRS-R 各子项 ≤ 2 的功能区域数
    regions = 0

    # 球部区域
    bulbar_items = [latest.speech, latest.salivation, latest.swallowing]
    if any(v is not None and v <= 2 for v in bulbar_items):
        regions += 1

    # 上肢区域
    upper_limb = [latest.handwriting, latest.cutting_food, latest.dressing_hygiene]
    if any(v is not None and v <= 2 for v in upper_limb):
        regions += 1

    # 下肢区域
    lower_limb = [latest.walking, latest.climbing_stairs, latest.turning_in_bed]
    if any(v is not None and v <= 2 for v in lower_limb):
        regions += 1

    # 呼吸区域
    resp_items = [latest.dyspnea, latest.orthopnea, latest.respiratory_insufficiency]
    if any(v is not None and v <= 2 for v in resp_items):
        regions += 1

    # 营养/呼吸衰竭 (Stage 4)
    from c9agent.data.patient_schema import MilestoneType
    has_gastrostomy = any(
        m.milestone_type == MilestoneType.GASTROSTOMY
        for m in patient.milestones
    )
    has_niv_dep = any(
        m.milestone_type == MilestoneType.NIV_DEPENDENCY
        for m in patient.milestones
    )
    if has_gastrostomy or has_niv_dep:
        return 4

    return min(regions, 4) if regions > 0 else 1


def calc_mitos_stage(patient: ALSPatientData) -> int | None:
    """
    MITOS 分期系统 (Milano-Torino Staging, Chiò et al., 2015)

    基于四个功能域的丧失:
    Stage 0 = 无丧失
    Stage 1 = 一个域丧失
    Stage 2 = 两个域丧失
    Stage 3 = 三个域丧失
    Stage 4 = 四个域丧失
    Stage 5 = 死亡

    功能域丧失定义:
    - 行走: walking ≤ 1
    - 上肢: cutting_food ≤ 1 AND dressing_hygiene ≤ 1
    - 吞咽: swallowing ≤ 1
    - 呼吸: respiratory_insufficiency ≤ 1 OR 使用 NIV
    """
    latest = patient.latest_alsfrsr
    if latest is None:
        return None

    losses = 0

    # 行走丧失
    if latest.walking is not None and latest.walking <= 1:
        losses += 1

    # 上肢丧失
    if (latest.cutting_food is not None and latest.cutting_food <= 1 and
            latest.dressing_hygiene is not None and latest.dressing_hygiene <= 1):
        losses += 1

    # 吞咽丧失
    if latest.swallowing is not None and latest.swallowing <= 1:
        losses += 1

    # 呼吸丧失
    has_niv = (patient.respiratory and
               patient.respiratory.niv_usage_hours_per_day is not None)
    if (latest.respiratory_insufficiency is not None and
            latest.respiratory_insufficiency <= 1) or has_niv:
        losses += 1

    return losses


def estimate_median_survival(
    onset_site: str,
    age_at_onset: float,
    alsfrsr_slope: float | None,
    fvc_percent: float | None,
    has_pathogenic_variant: bool = False,
) -> dict:
    """
    基于文献公式估算中位生存期。

    使用多个预后因子做粗略估算：
    - 球部起病: -6个月
    - 年龄 > 60: -6个月
    - 快速进展 (>0.89): -12个月
    - FVC < 70%: -6个月
    - 致病变异: -3到-12个月（取决于基因）

    注意: 这是简化估算，实际预测需要 CoxPH 等统计模型。

    Returns:
        dict with "median_months", "lower_ci", "upper_ci", "risk_level"
    """
    # ALS 中位生存期基线: ~30个月（从症状 onset 开始）
    base = 30.0

    adjustments = {}

    if onset_site == "bulbar":
        base -= 6
        adjustments["bulbar_onset"] = -6

    if age_at_onset > 60:
        base -= 6
        adjustments["age_gt_60"] = -6

    if alsfrsr_slope is not None:
        if alsfrsr_slope > 0.89:
            base -= 12
            adjustments["fast_progression"] = -12
        elif alsfrsr_slope > 0.45:
            base -= 4
            adjustments["moderate_progression"] = -4

    if fvc_percent is not None and fvc_percent < 70:
        base -= 6
        adjustments["low_fvc"] = -6

    if has_pathogenic_variant:
        base -= 6
        adjustments["pathogenic_variant"] = -6

    # 下限为3个月
    base = max(base, 3.0)

    # 粗略的 95% CI (± 40%)
    ci = base * 0.4

    if base < 12:
        risk = "high"
    elif base < 24:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "median_months": round(base, 1),
        "lower_ci": round(max(base - ci, 1), 1),
        "upper_ci": round(base + ci, 1),
        "risk_level": risk,
        "adjustments": adjustments,
        "note": "简化估算，非精确预测。精确预测请使用 survival_models.py 中的 CoxPH/DeepSurv 模型。"
    }
