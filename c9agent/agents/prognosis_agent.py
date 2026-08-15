"""
c9agent/agents/prognosis_agent.py — 预后预测 Agent

对应 DeepRare 的 Phenotype Analyzer + Genotype Analyzer 综合输出。
职责:
1. 运行生存预测模型（CoxPH 基线）
2. 估算关键里程碑时间
3. 综合基因型信息调整预测
"""
import time
from c9agent.data.patient_schema import ALSPatientData, MilestoneType
from c9agent.agents.base_agent import BaseAgent, AgentResult
from c9agent.models.survival_models import CoxPHSurvival, SurvivalPrediction
from c9agent.utils.als_calculators import (
    calc_alsfrsr_slope, classify_progression,
    calc_kings_stage, calc_mitos_stage, estimate_median_survival,
)


class PrognosisAgent(BaseAgent):
    """
    预后预测 Agent —— 预测生存期 + 里程碑时间。

    工作流:
    1. 运行 CoxPH 模型获取中位生存期和生存概率曲线
    2. 根据 ALSFRS-R 下降速率估算里程碑时间
    3. 综合基因型信息给出风险分层
    """

    def __init__(self):
        super().__init__(
            name="PrognosisAgent",
            description="生存预测 + ALSFRS-R 轨迹 + 里程碑估算",
        )
        self.survival_model = CoxPHSurvival()

    def execute(self, patient: ALSPatientData, **kwargs) -> AgentResult:
        t0 = time.time()
        warnings = []

        # —— Step 1: 生存预测 ——
        if patient.alsfrsr_slope is not None:
            surv_pred = self.survival_model.predict_from_patient(patient)
        else:
            # 没有 ALSFRS-R 记录，用简化公式
            surv_pred = self._fallback_prediction(patient)
            warnings.append("无 ALSFRS-R 记录，使用简化公式估算生存期")

        # —— Step 2: 里程碑时间估算 ——
        milestones = self._estimate_milestones(patient, surv_pred)

        # —— Step 3: 进展速度分类 ——
        slope = patient.alsfrsr_slope
        progression = classify_progression(slope) if slope else "unknown"

        # —— Step 4: 分期 ——
        kings = calc_kings_stage(patient)
        mitos = calc_mitos_stage(patient)

        # —— 构造证据节点 ——
        self._add_evidence(
            "inference",
            f"生存预测: 中位{surv_pred.median_survival_months}个月, "
            f"风险等级={surv_pred.risk_level}, "
            f"进展速度={progression}",
            source="CoxPH model (PRO-ACT, N=3220)",
            confidence=0.70 if patient.alsfrsr_slope else 0.40,
        )

        # 基因型调整
        if patient.genetic_variants:
            pathogenic = [v for v in patient.genetic_variants
                         if v.acmg_classification and
                         v.acmg_classification.value in ("P", "LP")]
            if pathogenic:
                genes = ", ".join(v.gene for v in pathogenic)
                self._add_evidence(
                    "evidence",
                    f"携带致病变异: {genes} —— 这可能意味着更快的进展",
                    source="ClinVar + ALS文献",
                    confidence=0.65,
                )

        elapsed = (time.time() - t0) * 1000

        return AgentResult(
            agent_name=self.name,
            status="success" if not warnings else "partial",
            data={
                "survival_prediction": {
                    "median_months": surv_pred.median_survival_months,
                    "survival_probabilities": surv_pred.survival_prob,
                    "risk_level": surv_pred.risk_level,
                    "confidence_interval": surv_pred.confidence_interval,
                },
                "milestones": milestones,
                "progression_rate": progression,
                "alsfrsr_slope": slope,
                "kings_stage": kings,
                "mitos_stage": mitos,
            },
            evidence_nodes=[],
            confidence=0.70 if patient.alsfrsr_slope else 0.40,
            warnings=warnings,
            execution_time_ms=elapsed,
        )

    def _estimate_milestones(
        self, patient: ALSPatientData, surv: SurvivalPrediction
    ) -> dict:
        """
        估算关键临床里程碑的时间。

        基于 ALSFRS-R 下降速率做线性外推，参考:
        - 丧失行走: walking ≤ 1
        - 丧失吞咽: swallowing ≤ 1
        - NIV 依赖: FVC < 50% 或 respiratory_insufficiency ≤ 1
        - 死亡: 中位生存期
        """
        slope = patient.alsfrsr_slope
        if slope is None:
            return {"note": "需要纵向ALSFRS-R数据来估算里程碑时间"}

        latest = patient.latest_alsfrsr
        milestones = {}

        # 从当前评分线性外推
        if latest and latest.walking is not None:
            months_to_walking_loss = max(0, (latest.walking - 1) / (slope / 4))
            milestones["loss_of_ambulation_months"] = round(months_to_walking_loss, 1)

        if latest and latest.swallowing is not None:
            months_to_swallow_loss = max(0, (latest.swallowing - 1) / (slope / 4))
            milestones["loss_of_swallowing_months"] = round(months_to_swallow_loss, 1)

        # 中位生存期（从发病起算的总月数）
        milestones["median_survival_from_onset_months"] = surv.median_survival_months

        return milestones

    def _fallback_prediction(self, patient: ALSPatientData) -> SurvivalPrediction:
        """无 ALSFRS-R 数据时的简化预后估算"""
        has_pathogenic = patient.has_pathogenic_variant
        fvc = (patient.respiratory.fvc_percent_predicted
               if patient.respiratory else None)

        est = estimate_median_survival(
            onset_site=patient.onset_site.value,
            age_at_onset=patient.age_at_onset,
            alsfrsr_slope=None,
            fvc_percent=fvc,
            has_pathogenic_variant=has_pathogenic,
        )

        from c9agent.models.survival_models import SurvivalPrediction
        return SurvivalPrediction(
            median_survival_months=est["median_months"],
            survival_prob={},
            risk_level=est["risk_level"],
            confidence_interval={"lower": est["lower_ci"], "upper": est["upper_ci"]},
        )
