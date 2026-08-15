"""
c9agent/core/reflection_loop.py — 自我反思循环

DeepRare 论文核心机制:
"the central host will make a tentative diagnosis, followed by a self-reflection
 phase in which it conducts additional searches to rigorously validate or refute
 these hypotheses. If no hypothesized diseases meet the self-reflection criteria,
 the system revisits earlier steps iteratively."

Phase 3 实现:
1. 从 Agent 结果 → 生成多个预后假设
2. 评估每个假设的证据强度
3. 识别知识缺口（缺失数据/不确定推论）
4. 补充查询 → 更新假设
5. 收敛或 3 次迭代后输出最终假设
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from c9agent.core.memory_bank import MemoryBank
from c9agent.utils.llm_client import run_llm


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class Hypothesis:
    """一个预后假设"""
    id: str
    statement: str                    # 假设陈述，如"该患者为快速进展型，中位生存~18个月"
    supporting_evidence: list[str] = field(default_factory=list)  # 支持的证据
    refuting_evidence: list[str] = field(default_factory=list)    # 反驳的证据
    confidence: float = 0.5           # 置信度 (0-1)
    source_agents: list[str] = field(default_factory=list)        # 来源 Agent
    iteration: int = 0                # 在哪次迭代中生成的


@dataclass
class KnowledgeGap:
    """一个知识缺口"""
    description: str                   # 缺失什么
    severity: str = "low"             # low / medium / high
    suggested_action: str = ""        # 建议的补全动作（如"重新查询FVC数据"）
    can_be_filled: bool = True        # 是否可以从现有数据中补全


@dataclass
class ReflectionResult:
    """反思循环的最终结果"""
    final_hypotheses: list[Hypothesis]
    total_iterations: int
    gaps_identified: list[KnowledgeGap]
    gaps_filled: int
    convergence_reached: bool
    reasoning_trace: str = ""          # 可追溯的推理过程文本


# ============================================================================
# 反思循环
# ============================================================================

class SelfReflectionLoop:
    """
    DeepRare 风格的自我反思循环。

    每轮迭代:
    1. 从 MemoryBank 读取所有 Agent 输出和累积证据
    2. LLM 生成预后假设（3-5个）
    3. LLM 评估每个假设的证据强度
    4. LLM 识别知识缺口
    5. 如果缺口可补 → 标记 re_query, Orchestrator 会重新调用相关 Agent
    6. 如果无法补 → 降低假设置信度, 标注不确定性
    7. 收敛检查 → 输出最终假设

    使用:
        loop = SelfReflectionLoop(llm_client=..., max_iterations=3)
        result = loop.run(memory, agents, patient)
    """

    MAX_HYPOTHESES = 5
    MAX_EVIDENCE_PER_HYPOTHESIS = 3

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self._history: list[list[Hypothesis]] = []  # 每轮迭代的假设历史

    def run(self, memory: MemoryBank,
            patient_summary: str,
            agent_outputs: str) -> ReflectionResult:
        """
        执行自我反思循环。

        参数:
            memory: 累积的 MemoryBank
            patient_summary: 患者信息摘要（文本）
            agent_outputs: 所有 Agent 输出的摘要（文本）

        返回:
            ReflectionResult 包含最终假设和反思追溯
        """
        hypotheses = []
        all_gaps = []

        for iteration in range(1, self.max_iterations + 1):
            # Step 1: 生成假设
            memory.add_reflection(f"迭代 {iteration}: 生成假设...")
            new_hypotheses = self._generate_hypotheses(
                patient_summary, agent_outputs,
                existing=hypotheses, iteration=iteration,
            )

            if not new_hypotheses:
                memory.add_reflection(f"迭代 {iteration}: 无法生成新假设，循环终止")
                break

            # Step 2: 评估每个假设
            for h in new_hypotheses:
                evaluation = self._evaluate_hypothesis(
                    h, agent_outputs, iteration,
                )
                h.supporting_evidence = evaluation.get("supporting", [])
                h.refuting_evidence = evaluation.get("refuting", [])
                h.confidence = evaluation.get("confidence", h.confidence)

            hypotheses = new_hypotheses

            # Step 3: 识别知识缺口
            gaps = self._identify_gaps(hypotheses, patient_summary)
            all_gaps.extend(gaps)

            if not gaps:
                memory.add_reflection(f"迭代 {iteration}: 无新缺口，收敛")
                break

            fillable = [g for g in gaps if g.can_be_filled]
            if not fillable:
                memory.add_reflection(
                    f"迭代 {iteration}: {len(gaps)} 个缺口无法从现有数据填充，"
                    f"标注不确定性"
                )
                break

            memory.add_reflection(
                f"迭代 {iteration}: 发现 {len(fillable)} 个可填充缺口，"
                f"需要补充查询"
            )

        # 构建反思追溯
        trace = self._build_trace(hypotheses, all_gaps)

        return ReflectionResult(
            final_hypotheses=hypotheses,
            total_iterations=iteration,
            gaps_identified=all_gaps,
            gaps_filled=len([g for g in all_gaps if g.can_be_filled]),
            convergence_reached=len(all_gaps) == 0 or iteration >= self.max_iterations,
            reasoning_trace=trace,
        )

    # —— 内部方法 ——

    def _generate_hypotheses(
        self, patient_summary: str, agent_outputs: str,
        existing: list[Hypothesis], iteration: int,
    ) -> list[Hypothesis]:
        """LLM 生成预后假设"""
        existing_text = "\n".join(
            f"- [{h.confidence:.0%}] {h.statement}" for h in existing
        ) if existing else "（无现有假设）"

        prompt = f"""你是一个 ALS 临床推理专家。基于以下患者数据和系统分析结果，
请生成 3-5 个具体的预后假设。每个假设应该是清晰、可验证的临床判断。

## 患者数据
{patient_summary[:2000]}

## 系统分析结果
{agent_outputs[:2000]}

## 现有假设（上一轮迭代）
{existing_text}

## 要求
请返回 JSON 格式（只返回 JSON）:
{{
  "hypotheses": [
    {{
      "statement": "具体的预后判断，包含关键数据支撑",
      "confidence": 0.0-1.0,
      "rationale": "支持该假设的关键推理"
    }}
  ]
}}

重要: 置信度必须基于数据证据强度。缺乏数据时使用较低置信度。"""
        try:
            answer = run_llm(prompt)
            data = json.loads(answer[answer.find("{"):answer.rfind("}")+1])
            hyps = []
            for i, h in enumerate(data.get("hypotheses", [])[:self.MAX_HYPOTHESES]):
                hyps.append(Hypothesis(
                    id=f"hyp_{iteration}_{i+1}",
                    statement=h.get("statement", ""),
                    confidence=max(0.1, min(0.95, h.get("confidence", 0.5))),
                    iteration=iteration,
                ))
            return hyps
        except (json.JSONDecodeError, ValueError):
            return []

    def _evaluate_hypothesis(
        self, hypothesis: Hypothesis, agent_outputs: str, iteration: int,
    ) -> dict:
        """LLM 评估一个假设的证据强度"""
        prompt = f"""你是一个 ALS 临床证据评估专家。评估以下预后假设的证据强度。

## 假设
{hypothesis.statement}

## 分析数据
{agent_outputs[:2000]}

## 要求
对这个假设，找出:
1. 支持证据（每条引用具体数据）
2. 反驳证据（如果有）
3. 综合置信度

返回 JSON:
{{
  "supporting": ["证据1", "证据2"],
  "refuting": ["反驳证据1"],
  "confidence": 0.0-1.0
}}"""
        try:
            answer = run_llm(prompt)
            return json.loads(answer[answer.find("{"):answer.rfind("}")+1])
        except (json.JSONDecodeError, ValueError):
            return {"supporting": [], "refuting": [], "confidence": hypothesis.confidence}

    def _identify_gaps(
        self, hypotheses: list[Hypothesis], patient_summary: str,
    ) -> list[KnowledgeGap]:
        """LLM 识别知识缺口"""
        hyps_text = "\n".join(
            f"- [{h.confidence:.0%}] {h.statement}"
            for h in hypotheses[:3]
        )

        prompt = f"""你是一个 ALS 临床数据分析师。基于以下假设和患者数据，识别知识缺口。

## 假设
{hyps_text}

## 患者数据
{patient_summary[:1500]}

## 要求
找出影响预测准确性的关键数据缺失或推理不确定处。

返回 JSON:
{{
  "gaps": [
    {{
      "description": "缺失什么信息",
      "severity": "high/medium/low",
      "can_be_filled": true/false,
      "suggested_action": "如何补全"
    }}
  ]
}}"""
        try:
            answer = run_llm(prompt)
            data = json.loads(answer[answer.find("{"):answer.rfind("}")+1])
            gaps = []
            for g in data.get("gaps", [])[:5]:
                gaps.append(KnowledgeGap(
                    description=g.get("description", ""),
                    severity=g.get("severity", "medium"),
                    can_be_filled=g.get("can_be_filled", False),
                    suggested_action=g.get("suggested_action", ""),
                ))
            return gaps
        except (json.JSONDecodeError, ValueError):
            return []

    def _build_trace(self, hypotheses: list[Hypothesis],
                     gaps: list[KnowledgeGap]) -> str:
        """构造可追溯的推理过程文本"""
        parts = ["# 临床推理过程\n"]

        parts.append("## 预后假设")
        for i, h in enumerate(hypotheses, 1):
            parts.append(f"\n### 假设 {i} (置信度: {h.confidence:.0%})")
            parts.append(h.statement)
            if h.supporting_evidence:
                parts.append("\n**支持证据:**")
                for ev in h.supporting_evidence[:3]:
                    parts.append(f"- {ev}")
            if h.refuting_evidence:
                parts.append("\n**反驳证据:**")
                for ev in h.refuting_evidence[:3]:
                    parts.append(f"- {ev}")

        if gaps:
            parts.append("\n## 知识缺口与不确定性")
            for g in gaps:
                parts.append(
                    f"- [{g.severity.upper()}] {g.description}"
                    f"{' (建议: ' + g.suggested_action + ')' if g.suggested_action else ''}"
                )

        return "\n".join(parts)
