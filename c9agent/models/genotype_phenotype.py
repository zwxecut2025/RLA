"""
c9agent/models/genotype_phenotype.py — 基因型-表型关联分析

提供队列级别的统计挖掘:
1. Kaplan-Meier 生存曲线（按基因型/起病部位分层）
2. 病例-对照关联检验（Fisher exact test）
3. 表型对比（突变型 vs 野生型的临床参数差异）

Phase 4: 用于在真实 ALS 队列中发现基因型-表型规律
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class KMSurvivalCurve:
    """Kaplan-Meier 生存曲线"""
    label: str                               # 曲线标签（如 "C9orf72+"）
    times_months: list[float]               # 时间点（月）
    survival_prob: list[float]              # 生存概率
    ci_lower: list[float]                   # 95% CI 下限
    ci_upper: list[float]                   # 95% CI 上限
    n_at_risk: list[int]                    # 各时间点的 at-risk 人数
    n_total: int                            # 总人数
    n_events: int                           # 事件数（死亡）
    median_survival: float = 0              # 中位生存期（月）


@dataclass
class AssociationResult:
    """基因-表型关联检验结果"""
    gene: str                                # 基因名
    phenotype: str                           # 表型（如"bulbar_onset", "fast_progression"）
    test: str                                # 检验方法
    odds_ratio: float = 1.0                  # 比值比
    p_value: float = 1.0                     # p 值
    ci_lower: float = 0                      # OR 95% CI
    ci_upper: float = 0
    n_mutation: int = 0                      # 突变组人数
    n_wildtype: int = 0                      # 野生型人数
    mutation_rate_in_phenotype: float = 0    # 表型+组中突变频率
    wildtype_rate_in_phenotype: float = 0    # 表型-组中突变频率
    significant: bool = False                # p < 0.05


@dataclass
class CohortComparison:
    """两组队列的临床参数比较"""
    group_name: str                          # "C9orf72+ vs C9orf72-"
    metrics: dict[str, dict] = None          # {metric: {group1_mean, group2_mean, p_value}}


# ============================================================================
# Kaplan-Meier 估计器（纯 Python，无外部依赖）
# ============================================================================

class KaplanMeierEstimator:
    """
    Kaplan-Meier 生存分析。

    使用方式:
        km = KaplanMeierEstimator()
        curve = km.fit(times, events, label="C9orf72+")
    """

    def fit(self, times: list[float], events: list[bool],
            label: str = "cohort") -> KMSurvivalCurve:
        """
        拟合 KM 生存曲线。

        参数:
            times: 生存时间（月）
            events: 事件指示（True=死亡, False=删失）
            label: 曲线标签
        """
        if not times:
            return KMSurvivalCurve(label=label, times_months=[],
                                   survival_prob=[], ci_lower=[],
                                   ci_upper=[], n_at_risk=[],
                                   n_total=0, n_events=0)

        # 排序
        sorted_data = sorted(zip(times, events), key=lambda x: x[0])
        sorted_t, sorted_e = zip(*sorted_data)

        unique_times = sorted(set(sorted_t))
        n = len(sorted_t)

        survival = [1.0]
        time_points = [0]
        n_at_risk = [n]
        ci_lower = [1.0]
        ci_upper = [1.0]

        for t in unique_times:
            # 计算该时间点的事件数和风险人数
            at_risk = sum(1 for ti in sorted_t if ti >= t)
            events_at_t = sum(1 for ti, ei in zip(sorted_t, sorted_e)
                            if ti == t and ei)

            if at_risk == 0:
                continue

            # 条件生存概率
            cond_surv = 1 - events_at_t / at_risk
            new_surv = survival[-1] * cond_surv

            # Greenwood 公式计算方差
            if at_risk > events_at_t:
                variance = new_surv**2 * events_at_t / (at_risk * (at_risk - events_at_t))
                se = np.sqrt(variance)
            else:
                se = 0

            survival.append(max(0, new_surv))
            time_points.append(t)
            n_at_risk.append(at_risk)

            # 95% CI (log-log transformation 避免越过 [0,1])
            z = 1.96
            ci_upper_val = min(1.0, new_surv * np.exp(z * se / new_surv)) if new_surv > 0 else 0
            ci_lower_val = max(0.0, new_surv * np.exp(-z * se / new_surv)) if new_surv > 0 else 0
            ci_lower.append(ci_lower_val)
            ci_upper.append(ci_upper_val)

        # 中位生存期
        median = 0
        for t, s in zip(time_points, survival):
            if s <= 0.5:
                median = t
                break

        return KMSurvivalCurve(
            label=label,
            times_months=time_points,
            survival_prob=survival,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_at_risk=n_at_risk,
            n_total=n,
            n_events=sum(1 for e in sorted_e if e),
            median_survival=median,
        )


def logrank_test(group1_times: list[float], group1_events: list[bool],
                 group2_times: list[float], group2_events: list[bool]) -> dict:
    """
    Log-rank 检验（两组生存曲线比较）。

    返回: {"chi2": float, "p_value": float}
    """
    from scipy.stats import chi2 as chi2_dist

    all_times = sorted(set(group1_times + group2_times))
    o1_sum, e1_sum = 0.0, 0.0

    for t in all_times:
        n1 = sum(1 for ti in group1_times if ti >= t)
        n2 = sum(1 for ti in group2_times if ti >= t)
        d1 = sum(1 for ti, ei in zip(group1_times, group1_events)
                if ti == t and ei)
        d2 = sum(1 for ti, ei in zip(group2_times, group2_events)
                if ti == t and ei)

        n_total = n1 + n2
        d_total = d1 + d2
        if n_total == 0 or d_total == 0:
            continue

        e1 = n1 * d_total / n_total
        o1_sum += d1
        e1_sum += e1

    chi2 = (o1_sum - e1_sum)**2 / max(0.001, e1_sum)
    p_value = 1 - chi2_dist.cdf(chi2, 1)

    return {"chi2": chi2, "p_value": p_value, "significant": p_value < 0.05}


# ============================================================================
# 关联检验
# ============================================================================

def fisher_exact_test(a: int, b: int, c: int, d: int) -> dict:
    """
    Fisher 精确检验（2x2 列联表）。

    表格式:
              表型+  表型-
      突变+    a      b
      突变-    c      d
    """
    from scipy.stats import fisher_exact
    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table)

    # OR 的 95% CI (Woolf 法取对数)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a,b,c,d) > 0 else float('inf')
    log_or = np.log(max(0.01, odds_ratio))
    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)

    return {
        "odds_ratio": round(odds_ratio, 3),
        "p_value": round(p_value, 6),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "significant": p_value < 0.05,
    }


def t_test(group1: list[float], group2: list[float]) -> dict:
    """独立样本 t 检验（Welch）"""
    from scipy.stats import ttest_ind
    stat, p = ttest_ind(group1, group2, equal_var=False)
    return {
        "mean1": round(np.mean(group1), 2),
        "mean2": round(np.mean(group2), 2),
        "t_statistic": round(stat, 3),
        "p_value": round(p, 6),
        "significant": p < 0.05,
    }


# ============================================================================
# 基因型-表型关联挖掘器
# ============================================================================

class GenotypePhenotypeAnalyzer:
    """
    队列级别的基因型-表型关联分析。

    使用:
        analyzer = GenotypePhenotypeAnalyzer()
        result = analyzer.analyze_gene("C9orf72", cohort, "bulbar_onset")
    """

    def analyze_gene_phenotype(self, gene: str, patients: list,
                                phenotype: str) -> AssociationResult:
        """
        分析特定基因与表型的关联。

        参数:
            gene: 基因名
            patients: ALSPatientData 列表
            phenotype: 表型名（"bulbar_onset", "fast_progression"）

        返回:
            AssociationResult
        """
        # 分层
        mutation_positive = [
            p for p in patients
            if any(v.gene == gene for v in p.genetic_variants)
        ]
        mutation_negative = [
            p for p in patients
            if not any(v.gene == gene for v in p.genetic_variants)
        ]

        if phenotype == "bulbar_onset":
            a = sum(1 for p in mutation_positive if p.is_bulbar_onset)
            b = sum(1 for p in mutation_positive if not p.is_bulbar_onset)
            c = sum(1 for p in mutation_negative if p.is_bulbar_onset)
            d = sum(1 for p in mutation_negative if not p.is_bulbar_onset)
        elif phenotype == "fast_progression":
            a = sum(1 for p in mutation_positive
                   if p.progression_category == "fast")
            b = sum(1 for p in mutation_positive
                   if p.progression_category != "fast")
            c = sum(1 for p in mutation_negative
                   if p.progression_category == "fast")
            d = sum(1 for p in mutation_negative
                   if p.progression_category != "fast")
        else:
            return AssociationResult(gene=gene, phenotype=phenotype,
                                     test="fisher", p_value=1.0)

        # Fisher 检验
        if min(a, b, c, d) > 0:
            result = fisher_exact_test(a, b, c, d)
        else:
            result = {"odds_ratio": 0, "p_value": 1.0,
                      "ci_lower": 0, "ci_upper": 0, "significant": False}

        return AssociationResult(
            gene=gene, phenotype=phenotype, test="fisher_exact",
            odds_ratio=result["odds_ratio"],
            p_value=result["p_value"],
            ci_lower=result["ci_lower"],
            ci_upper=result["ci_upper"],
            n_mutation=len(mutation_positive),
            n_wildtype=len(mutation_negative),
            mutation_rate_in_phenotype=(a / len(mutation_positive)
                                        if mutation_positive else 0),
            wildtype_rate_in_phenotype=(c / len(mutation_negative)
                                        if mutation_negative else 0),
            significant=result["significant"],
        )

    def compare_survival_by_gene(self, gene: str, patients: list,
                                  survival_times: dict[str, float],
                                  events: dict[str, bool]) -> dict:
        """
        按基因分层的生存曲线比较。

        返回: {"km_positive": KMSurvivalCurve, "km_negative": KMSurvivalCurve,
                "logrank_p": float}
        """
        pos_ids = {
            p.patient_id for p in patients
            if any(v.gene == gene for v in p.genetic_variants)
        }

        pos_times = [survival_times[pid] for pid in pos_ids
                     if pid in survival_times]
        pos_events = [events.get(pid, True) for pid in pos_ids
                      if pid in survival_times]
        neg_times = [survival_times[pid] for pid in survival_times
                     if pid not in pos_ids]
        neg_events = [events.get(pid, True) for pid in survival_times
                      if pid not in pos_ids]

        km = KaplanMeierEstimator()
        curve_pos = km.fit(pos_times, pos_events, f"{gene}+")
        curve_neg = km.fit(neg_times, neg_events, f"{gene}-")

        logrank = logrank_test(pos_times, pos_events, neg_times, neg_events)

        return {
            "km_positive": curve_pos,
            "km_negative": curve_neg,
            "logrank_p": logrank["p_value"],
            "significant": logrank["significant"],
            "median_pos": curve_pos.median_survival,
            "median_neg": curve_neg.median_survival,
        }

    def compare_clinical_parameters(self, gene: str,
                                     patients: list) -> CohortComparison:
        """比较突变型 vs 野生型的临床参数差异"""
        pos = [p for p in patients
               if any(v.gene == gene for v in p.genetic_variants)]
        neg = [p for p in patients
               if not any(v.gene == gene for v in p.genetic_variants)]

        metrics = {}

        # 年龄
        ages_pos = [p.age_at_onset for p in pos]
        ages_neg = [p.age_at_onset for p in neg]
        metrics["age_at_onset"] = t_test(ages_pos, ages_neg)

        # ALSFRS-R 下降速率
        slopes_pos = [p.alsfrsr_slope for p in pos if p.alsfrsr_slope]
        slopes_neg = [p.alsfrsr_slope for p in neg if p.alsfrsr_slope]
        if slopes_pos and slopes_neg:
            metrics["alsfrsr_slope"] = t_test(slopes_pos, slopes_neg)

        return CohortComparison(
            group_name=f"{gene}+ (n={len(pos)}) vs {gene}- (n={len(neg)})",
            metrics=metrics,
        )
