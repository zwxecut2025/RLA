"""
c9agent/core/report_builder.py — 可追溯推理报告生成器

DeepRare 核心输出:
"DeepRare outputs a ranked list of potential rare diseases, each accompanied
 by a transparent reasoning chain that links each inference step directly to
 trusted medical evidence."

每个报告的推理链是一组 EvidenceNode，每个节点有:
- type: observation / inference / hypothesis / evidence / conclusion
- source: 证据来源 (PMID / ClinVar ID / 模型版本 / Agent 名称)
- confidence: 置信度
- parent_ids: 上游推理节点

报告格式: JSON（机器可读）+ Markdown（人类可读）
"""

import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from c9agent.data.knowledge.knowledge_base import KnowledgeBase


# ============================================================================
# 推理链节点
# ============================================================================

@dataclass
class EvidenceNode:
    """推理链中的一个节点"""
    node_id: str                                              # 唯一 ID
    type: str                                                 # observation|inference|hypothesis|evidence|conclusion
    content: str                                              # 节点内容
    source: Optional[str] = None                              # PMID / ClinVar ID / 模型版本
    source_url: Optional[str] = None                          # 可点击的链接
    confidence: float = 1.0                                   # 0-1
    parent_ids: list[str] = field(default_factory=list)       # 链接到上游节点
    iteration: int = 0                                        # 在反思循环的哪次迭代中生成

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "type": self.type,
            "content": self.content,
            "source": self.source,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "parent_ids": self.parent_ids,
            "iteration": self.iteration,
        }

    def to_markdown(self) -> str:
        """单节点 Markdown 格式"""
        icons = {
            "observation": "📋",
            "inference": "🧠",
            "hypothesis": "💡",
            "evidence": "📚",
            "conclusion": "✅",
        }
        icon = icons.get(self.type, "•")
        md = f"{icon} **[{self.type.upper()}]** {self.content}"
        if self.source:
            md += f"\n  *来源: {self.source}*"
            if self.source_url:
                md += f" — [链接]({self.source_url})"
        md += f"\n  *置信度: {self.confidence:.0%}*"
        return md


# ============================================================================
# 报告生成器
# ============================================================================

class TraceableReportBuilder:
    """
    可追溯报告生成器。

    输入: 患者数据 + Agent 结果 + 反思循环输出
    输出: 结构化的 AnalysisReport（含推理链 DAG）
    """

    def __init__(self):
        self._reasoning_nodes: list[EvidenceNode] = []
        self._node_counter = 0

    def add_node(self, node_type: str, content: str,
                 source: str = None, source_url: str = None,
                 confidence: float = 1.0,
                 parent_ids: list[str] = None,
                 iteration: int = 0) -> EvidenceNode:
        """添加一个推理节点"""
        self._node_counter += 1
        node = EvidenceNode(
            node_id=f"node_{self._node_counter:04d}",
            type=node_type,
            content=content,
            source=source,
            source_url=source_url,
            confidence=confidence,
            parent_ids=parent_ids or [],
            iteration=iteration,
        )
        self._reasoning_nodes.append(node)
        return node

    def build(self, patient,
              agent_results: dict,
              reflection_result=None,
              memory=None) -> dict:
        """
        构建完整的分析报告。

        参数:
            patient: ALSPatientData
            agent_results: {agent_name: AgentResult}
            reflection_result: ReflectionResult (Phase 3)
            memory: MemoryBank

        返回:
            完整报告字典
        """
        nodes = list(self._reasoning_nodes)

        # 从 MemoryBank 合并推理链
        if memory:
            for raw in memory.get_reasoning_chain():
                node = EvidenceNode(
                    node_id=raw.get("node_id", f"mem_{len(nodes)}"),
                    type=raw.get("type", "inference"),
                    content=raw.get("content", ""),
                    source=raw.get("source"),
                    confidence=raw.get("confidence", 0.5),
                )
                nodes.append(node)

        # 从反思循环加入假设
        if reflection_result:
            for hyp in reflection_result.final_hypotheses:
                self.add_node(
                    "hypothesis",
                    hyp.statement,
                    confidence=hyp.confidence,
                    iteration=hyp.iteration,
                )

        report = {
            "report_type": "ALS Individualized Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "patient_summary": self._build_patient_summary(patient),
            "prognosis": self._build_prognosis(agent_results),
            "literature_evidence": self._build_literature(agent_results),
            "knowledge_base": self._build_knowledge_base(patient),
            "genetics": self._build_genetics(patient),
            "reflection": self._build_reflection(reflection_result),
            "reasoning_chain": [n.to_dict() for n in nodes],
            "confidence_assessment": self._build_confidence(patient, agent_results, reflection_result),
            "warnings": self._collect_warnings(agent_results),
        }

        return report

    def to_markdown(self, report: dict) -> str:
        """JSON 报告 → Markdown 文本"""
        md = []
        md.append("# ALS 个体化分析报告")
        md.append(f"*生成时间: {report['generated_at']}*")

        # 患者概要
        ps = report["patient_summary"]
        md.append("\n## 1. 患者概要")
        md.append(f"- 患者: {ps.get('patient_id', 'N/A')}")
        md.append(f"- 性别: {ps.get('sex')}, 发病年龄: {ps.get('age_at_onset')}岁")
        md.append(f"- 起病部位: {ps.get('onset_site')}")
        md.append(f"- 最新 ALSFRS-R: {ps.get('latest_alsfrsr')}/48")
        if ps.get("alsfrsr_slope"):
            md.append(f"- 下降速率: {ps['alsfrsr_slope']:.2f} 分/月 ({ps.get('progression_category', '')})")
        md.append(f"- King's 分期: Stage {ps.get('kings_stage', 'N/A')}")

        # 预后
        prog = report.get("prognosis", {})
        surv = prog.get("survival", {}) or {}
        md.append("\n## 2. 预后预测")
        md.append(f"- 中位生存期: **{surv.get('median_months', 'N/A')} 个月**")
        md.append(f"- 风险等级: **{surv.get('risk_level', 'N/A').upper()}**")
        if surv.get("confidence_interval"):
            ci = surv["confidence_interval"]
            md.append(f"- 95% CI: {ci.get('lower')} - {ci.get('upper')} 个月")
        if surv.get("survival_probabilities"):
            md.append("- 生存概率:")
            for months, prob in surv["survival_probabilities"].items():
                if int(months) in [12, 24, 36]:
                    md.append(f"  - {months}月: {prob*100:.0f}%")

        # 文献证据
        lit = report.get("literature_evidence", {}) or {}
        articles = lit.get("articles", [])
        if articles:
            md.append("\n## 3. 文献检索")
            for a in articles[:5]:
                md.append(f"- [{a.get('year')} | {a.get('journal')}] **{a.get('title', '')[:100]}**")
                md.append(f"  PMID:{a.get('pmid')} — {a.get('url')}")
                for ev in a.get("evidence_statements", [])[:2]:
                    md.append(f"  > {ev[:150]}")

        # 知识库参考
        kb = report.get("knowledge_base", {}) or {}
        kb_items = kb.get("items", [])
        feature_contrib = kb.get("feature_contributions", [])
        if kb_items or feature_contrib:
            md.append("\n## 4. 知识库参考")
            md.append(f"*本地知识库版本: {kb.get('kb_updated', 'N/A')}, 共 {kb.get('total_items', 0)} 条*")

            # 预测依据（特征贡献分析）
            if feature_contrib:
                md.append("\n### 4.1 预测依据（特征贡献分解）")
                md.append("以下展示每个特征对本次生存预测的具体贡献（β × 特征值 = 贡献值）：")
                md.append("| 特征 | 系数(β) | 实际值 | 贡献值 | 方向 |")
                md.append("|------|---------|--------|--------|------|")
                for fc in feature_contrib:
                    direction = "⬆ 增风险" if fc["contribution"] > 0 else "⬇ 减风险"
                    md.append(f"| {fc['feature']} | {fc['beta']:.3f} | {fc['value']:.2f} | {fc['contribution']:+.3f} | {direction} |")
                md.append(f"| **合计 (log-HR)** | | | **{sum(fc['contribution'] for fc in feature_contrib):+.3f}** | |")
                md.append(f"| **风险比 (HR)** | | | **{kb.get('hazard_ratio', 'N/A')}** | |")
                md.append("")

            # 相关知识
            if kb_items:
                md.append("\n### 4.2 相关临床知识")
                for item in kb_items[:5]:
                    md.append(f"- **{item.get('title', '')}** [{item.get('category', '')}]")
                    if item.get("summary"):
                        md.append(f"  {item.get('summary', '')[:200]}")
                    refs = item.get("references", [])
                    for r in refs[:1]:
                        r_text = r if isinstance(r, str) else r.get("citation", "")
                        md.append(f"  > 文献: {r_text[:150]}")

        # 推理链
        chain = report.get("reasoning_chain", [])
        if chain:
            md.append("\n## 5. 临床推理链")
            for node in chain:
                en = EvidenceNode(**node)
                md.append(en.to_markdown())
                md.append("")

        # 反思
        refl = report.get("reflection", {}) or {}
        if refl.get("hypotheses"):
            md.append("\n## 6. 自我反思")
            md.append(f"*迭代次数: {refl.get('total_iterations', 0)}*")
            for h in refl.get("hypotheses", []):
                md.append(f"- [{h.get('confidence', 0):.0%}] {h.get('statement', '')}")

        # 不确定性
        conf = report.get("confidence_assessment", {}) or {}
        md.append("\n## 7. 置信度评估")
        md.append(f"- 数据完整度: {conf.get('data_completeness', 'N/A')}")
        md.append(f"- 预测置信度: {conf.get('prediction_confidence', 'N/A')}")
        if conf.get("limitations"):
            md.append("- 局限性:")
            for lim in conf["limitations"]:
                md.append(f"  - {lim}")

        warnings = report.get("warnings", [])
        if warnings:
            md.append("\n## 8. 注意事项")
            for w in warnings:
                md.append(f"- ⚠ {w}")

        return "\n".join(md)

    # —— 构建各章节 ——

    def _build_patient_summary(self, patient) -> dict:
        return {
            "patient_id": patient.patient_id,
            "sex": patient.sex.value,
            "age_at_onset": patient.age_at_onset,
            "onset_site": patient.onset_site.value,
            "diagnosis_date": patient.diagnosis_date.isoformat(),
            "diagnostic_delay_months": round(patient.diagnostic_delay_months, 1),
            "latest_alsfrsr": patient.latest_alsfrsr.total_score if patient.latest_alsfrsr else None,
            "alsfrsr_slope": round(patient.alsfrsr_slope, 2) if patient.alsfrsr_slope else None,
            "progression_category": patient.progression_category,
            "kings_stage": patient.kings_stage,
        }

    def _build_prognosis(self, agent_results: dict) -> dict:
        prog = agent_results.get("PrognosisAgent")
        if prog and prog.status != "failed":
            return {
                "survival": prog.data.get("survival_prediction"),
                "milestones": prog.data.get("milestones", {}),
            }
        return {}

    def _build_literature(self, agent_results: dict) -> dict:
        lit = agent_results.get("LiteratureAgent")
        if lit and lit.status != "failed":
            return lit.data
        return {}

    def _build_genetics(self, patient) -> dict:
        return {
            "has_pathogenic_variant": patient.has_pathogenic_variant,
            "variants": [
                {"gene": v.gene, "type": v.variant_type.value,
                 "acmg": v.acmg_classification.value if v.acmg_classification else None}
                for v in patient.genetic_variants
            ],
            "family_history": patient.family_history_als,
        }

    def _build_knowledge_base(self, patient) -> dict:
        """从本地知识库检索相关知识并计算特征贡献"""
        try:
            kb = KnowledgeBase()
        except Exception:
            return {}

        # 获取患者相关文档
        genes = [v.gene for v in (patient.genetic_variants or [])]
        ctx = kb.get_context_for_patient(
            onset_site=patient.onset_site.value if patient.onset_site else "unknown",
            progression=patient.progression_category or "unknown",
            genes=genes,
            fvc=patient.respiratory.fvc_percent_predicted if patient.respiratory else None,
            age=patient.age_at_onset,
            max_items=6,
        )

        items = []
        for c in ctx:
            doc = kb.get_document(c["topic_id"])
            items.append({
                "topic_id": c["topic_id"],
                "title": c["title"],
                "category": c["category"],
                "summary": c.get("summary", ""),
                "references": c.get("references", []) if c.get("references") else
                              (doc["meta"].get("references", []) if doc else []),
            })

        # 计算特征贡献
        feature_contributions = self._calc_feature_contributions(patient)

        # 计算 HR
        import math
        total_log_hr = sum(fc["contribution"] for fc in feature_contributions)
        hr = round(math.exp(total_log_hr), 2)

        return {
            "items": items,
            "total_items": len(items),
            "feature_contributions": feature_contributions,
            "hazard_ratio": hr,
            "kb_updated": kb._index_data.get("updated", "") if kb._index_data else "",
        }

    def _calc_feature_contributions(self, patient) -> list[dict]:
        """计算每个特征对预测的贡献值（β × X）"""
        import json
        from pathlib import Path

        # 加载模型系数
        coef_path = Path(__file__).parent.parent.parent / "data" / "model_coefficients.json"
        sources = {}
        try:
            with open(coef_path, "r", encoding="utf-8") as f:
                coef_data = json.load(f)
            sources = coef_data.get("_coefficient_sources", {})
        except Exception:
            pass

        coef = {
            "age_per_10yr": 0.296,
            "bulbar_onset": 0.300,
            "alsfrsr_slope_per_point": 0.392,
            "fvc_per_10pct": -0.121,
            "diagnostic_delay_per_month": -0.0253,
            "c9orf72_positive": 0.372,
            "sod1_positive": 0.182,
            "male": 0.049,
        }

        # 计算特征值
        slope = patient.alsfrsr_slope or 0.5
        fvc_val = (patient.respiratory.fvc_percent_predicted if patient.respiratory else 100) or 100
        has_c9 = any(v.gene == "C9orf72" and v.acmg_classification and
                     v.acmg_classification.value in ("P", "LP")
                     for v in (patient.genetic_variants or []))
        has_sod1 = any(v.gene == "SOD1" and v.acmg_classification and
                       v.acmg_classification.value in ("P", "LP")
                       for v in (patient.genetic_variants or []))

        features = [
            ("年龄(per 10yr)", coef["age_per_10yr"], max(0, patient.age_at_onset - 60) / 10.0),
            ("球部起病", coef["bulbar_onset"], 1.0 if patient.is_bulbar_onset else 0.0),
            ("ALSFRS-R下降速率", coef["alsfrsr_slope_per_point"], slope),
            ("FVC(per 10%)", coef["fvc_per_10pct"], (fvc_val - 100) / 10.0),
            ("诊断延迟(月)", coef["diagnostic_delay_per_month"], patient.diagnostic_delay_months or 0),
            ("C9orf72突变", coef["c9orf72_positive"], 1.0 if has_c9 else 0.0),
            ("SOD1突变", coef["sod1_positive"], 1.0 if has_sod1 else 0.0),
            ("男性", coef["male"], 1.0 if patient.sex.value == "male" else 0.0),
        ]

        contributions = []
        for name, beta, value in features:
            contrib = beta * value
            contributions.append({
                "feature": name,
                "beta": round(beta, 3),
                "value": round(value, 3),
                "contribution": round(contrib, 3),
                "literature_hr": sources.get(name, {}).get("literature_hr", "") if isinstance(sources.get(name), dict) else "",
                "source_ref": sources.get(name, {}).get("source", "") if isinstance(sources.get(name), dict) else "",
            })

        # 按贡献绝对值降序排列
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions

    def _build_reflection(self, reflection_result) -> dict:
        if not reflection_result:
            return {}
        return {
            "total_iterations": reflection_result.total_iterations,
            "convergence_reached": reflection_result.convergence_reached,
            "gaps_identified": len(reflection_result.gaps_identified),
            "hypotheses": [
                {"statement": h.statement, "confidence": h.confidence}
                for h in reflection_result.final_hypotheses
            ],
            "reasoning_trace": reflection_result.reasoning_trace,
        }

    def _build_confidence(self, patient, agent_results, reflection) -> dict:
        """综合置信度评估"""
        data_completeness = []
        if patient.alsfrsr_records:
            data_completeness.append("ALSFRS-R 记录")
        if patient.respiratory and patient.respiratory.fvc_percent_predicted:
            data_completeness.append("FVC 数据")
        if patient.genetic_variants:
            data_completeness.append("基因检测")
        if patient.medications:
            data_completeness.append("用药记录")

        completeness_score = len(data_completeness) / 5.0  # 5类数据
        prediction_conf = 0.70
        if reflection and reflection.final_hypotheses:
            avg_conf = sum(h.confidence for h in reflection.final_hypotheses) / len(reflection.final_hypotheses)
            prediction_conf = avg_conf

        limitations = []
        if not patient.alsfrsr_records:
            limitations.append("缺少 ALSFRS-R 纵向数据，无法精确计算进展速率")
        if not patient.genetic_variants:
            limitations.append("无基因检测结果，可能遗漏 C9orf72/SOD1 等致病变异")
        if not patient.respiratory or not patient.respiratory.fvc_percent_predicted:
            limitations.append("缺少 FVC 数据，呼吸预后不确定性高")

        return {
            "data_completeness": f"{len(data_completeness)}/5 ({completeness_score:.0%})",
            "prediction_confidence": f"{prediction_conf:.0%}",
            "limitations": limitations,
        }

    def _collect_warnings(self, agent_results: dict) -> list[str]:
        warnings = []
        for r in agent_results.values():
            warnings.extend(r.warnings)
        return warnings
