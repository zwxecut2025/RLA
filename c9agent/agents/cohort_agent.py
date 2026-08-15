"""
c9agent/agents/cohort_agent.py — 队列分析 Agent

从单患者 → 人群级别挖掘:
1. 队列描述性统计（年龄/起病/进展/生存分布）
2. 基因型-表型关联分析
3. Kaplan-Meier 分层生存曲线
4. 亚组比较
"""

import time
import numpy as np
from c9agent.data.patient_schema import ALSPatientData
from c9agent.agents.base_agent import BaseAgent, AgentResult
from c9agent.models.survival_models import CoxPHSurvival
from c9agent.models.genotype_phenotype import (
    GenotypePhenotypeAnalyzer, KaplanMeierEstimator,
    AssociationResult, CohortComparison, logrank_test,
)
from c9agent.config import ALS_GENES


class CohortAgent(BaseAgent):
    """
    队列分析 Agent —— 人群级别的统计分析。

    使用:
        agent = CohortAgent()
        result = agent.execute(cohort=patients, survival_data={...})
    """

    def __init__(self):
        super().__init__(
            name="CohortAgent",
            description="队列统计分析: 基因型-表型关联 + KM 生存曲线 + 亚组比较",
        )
        self.genotype_analyzer = GenotypePhenotypeAnalyzer()
        self.km = KaplanMeierEstimator()
        self.survival_model = CoxPHSurvival()

    def execute(self, patient: ALSPatientData = None,
                cohort: list[ALSPatientData] = None,
                survival_data: dict[str, float] = None,
                events: dict[str, bool] = None,
                **kwargs) -> AgentResult:
        """
        执行队列分析。

        参数:
            cohort: 患者列表
            survival_data: {patient_id: survival_months}
            events: {patient_id: True=死亡/False=删失}
        """
        if not cohort:
            return AgentResult.failed(self.name, "未提供队列数据")

        t0 = time.time()
        n = len(cohort)

        # —— 1. 队列描述统计 ——
        desc = self._describe_cohort(cohort)

        # —— 2. 基因型-表型关联 ——
        associations = []
        for gene in [g for g in ALS_GENES
                     if any(any(v.gene == g for v in p.genetic_variants)
                           for p in cohort)]:
            for phenotype in ["bulbar_onset", "fast_progression"]:
                result = self.genotype_analyzer.analyze_gene_phenotype(
                    gene, cohort, phenotype
                )
                if result.significant:
                    associations.append(result)

        # —— 3. KM 总体生存曲线 ——
        if survival_data:
            times = [survival_data[p.patient_id]
                    for p in cohort if p.patient_id in survival_data]
            evts = [events.get(p.patient_id, True)
                   for p in cohort if p.patient_id in survival_data]
            overall_km = self.km.fit(times, evts, "All ALS Patients")
        else:
            overall_km = None

        # —— 4. 按起病部位分层 KM ——
        stratified = {}
        if survival_data:
            for group_name, filter_fn in [
                ("Bulbar Onset", lambda p: p.is_bulbar_onset),
                ("Spinal Onset", lambda p: not p.is_bulbar_onset),
            ]:
                group = [p for p in cohort if filter_fn(p)]
                g_times = [survival_data[p.patient_id]
                          for p in group if p.patient_id in survival_data]
                g_events = [events.get(p.patient_id, True)
                           for p in group if p.patient_id in survival_data]
                if g_times:
                    stratified[group_name] = self.km.fit(
                        g_times, g_events, group_name
                    )

        # —— 5. 基因型-表型临床参数比较 ——
        comparisons = []
        for gene in [g for g in ALS_GENES[:5]
                     if any(any(v.gene == g for v in p.genetic_variants)
                           for p in cohort)]:
            comp = self.genotype_analyzer.compare_clinical_parameters(
                gene, cohort
            )
            comparisons.append(comp)

        # —— 6. 写入证据 ——
        self._add_evidence("observation",
            f"队列分析: {n}例患者, 中位年龄={desc['mean_age']}岁, "
            f"球部起病={desc['bulbar_pct']:.0f}%, "
            f"中位生存={desc['median_survival']}月",
            confidence=0.95,
        )

        for assoc in associations[:5]:
            self._add_evidence("evidence",
                f"{assoc.gene} × {assoc.phenotype}: "
                f"OR={assoc.odds_ratio:.2f}, p={assoc.p_value:.4f}",
                confidence=0.80,
            )

        elapsed = (time.time() - t0) * 1000

        return AgentResult(
            agent_name=self.name,
            status="success",
            data={
                "cohort_size": n,
                "description": desc,
                "overall_survival": self._km_to_dict(overall_km) if overall_km else None,
                "stratified_survival": {
                    k: self._km_to_dict(v) for k, v in stratified.items()
                },
                "gene_phenotype_associations": [
                    {
                        "gene": a.gene, "phenotype": a.phenotype,
                        "odds_ratio": a.odds_ratio, "p_value": a.p_value,
                        "significant": a.significant,
                        "n_mutation": a.n_mutation, "n_wildtype": a.n_wildtype,
                    }
                    for a in associations
                ],
                "clinical_comparisons": [
                    {
                        "group": c.group_name,
                        "metrics": {k: v for k, v in c.metrics.items()},
                    }
                    for c in comparisons
                ],
            },
            evidence_nodes=[],
            confidence=0.90,
            execution_time_ms=elapsed,
        )

    def _describe_cohort(self, cohort: list[ALSPatientData]) -> dict:
        """队列描述性统计"""
        ages = [p.age_at_onset for p in cohort]
        slopes = [p.alsfrsr_slope for p in cohort if p.alsfrsr_slope]
        bulbar = sum(1 for p in cohort if p.is_bulbar_onset)
        pathogenic = sum(1 for p in cohort if p.has_pathogenic_variant)
        males = sum(1 for p in cohort if p.sex.value == "male")

        # 进展分类
        fast = sum(1 for p in cohort if p.progression_category == "fast")
        slow = sum(1 for p in cohort if p.progression_category == "slow")

        # 基因频数
        gene_counts = {}
        for gene in ALS_GENES:
            cnt = sum(1 for p in cohort
                     if any(v.gene == gene for v in p.genetic_variants))
            if cnt > 0:
                gene_counts[gene] = cnt

        return {
            "total": len(cohort),
            "mean_age": round(np.mean(ages), 1),
            "std_age": round(np.std(ages), 1),
            "min_age": min(ages),
            "max_age": max(ages),
            "mean_slope": round(np.mean(slopes), 2) if slopes else None,
            "std_slope": round(np.std(slopes), 2) if slopes else None,
            "bulbar_count": bulbar,
            "bulbar_pct": round(bulbar / len(cohort) * 100, 1),
            "male_count": males,
            "male_pct": round(males / len(cohort) * 100, 1),
            "pathogenic_variant_pct": round(pathogenic / len(cohort) * 100, 1),
            "fast_progression_pct": round(fast / len(cohort) * 100, 1),
            "slow_progression_pct": round(slow / len(cohort) * 100, 1),
            "gene_counts": gene_counts,
            "median_survival": None,  # 需要 survival_data
        }

    def _km_to_dict(self, curve) -> dict:
        """KM 曲线 → 可序列化的字典"""
        if curve is None:
            return None
        return {
            "label": curve.label,
            "n_total": curve.n_total,
            "n_events": curve.n_events,
            "median_survival": curve.median_survival,
            "times_months": curve.times_months[:20],    # 截断避免过大
            "survival_prob": [round(s, 3) for s in curve.survival_prob[:20]],
            "ci_lower": [round(c, 3) for c in curve.ci_lower[:20]],
            "ci_upper": [round(c, 3) for c in curve.ci_upper[:20]],
        }
