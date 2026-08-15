"""
c9agent/core/orchestrator.py — 中央调度器 (Central Host)

这是整个系统的"大脑"，对应 DeepRare 论文的 Central Host:
"The central host decomposes the diagnostic task systematically. It first
 orchestrates the agent servers to retrieve relevant evidence, then synthesizes
 this evidence to generate hypotheses, followed by a self-reflection phase."

Phase 1 版本: 基本的串行 Agent 调度 + 证据合成 + 报告生成
Phase 3 将加入: SelfReflectionLoop（自我反思循环）

工作流:
1. PhenotypeAgent: 规范化输入
2. [并行] PrognosisAgent: 生存预测
3. Evidence Synthesis (LLM): 合成证据
4. ReportBuilder: 生成可追溯报告
"""

import time
from datetime import datetime
from c9agent.data.patient_schema import ALSPatientData, ALSPatientInput
from c9agent.core.memory_bank import MemoryBank
from c9agent.core.reflection_loop import SelfReflectionLoop
from c9agent.core.report_builder import TraceableReportBuilder
from c9agent.agents.base_agent import BaseAgent, AgentResult
from c9agent.agents.phenotype_agent import PhenotypeAgent
from c9agent.agents.prognosis_agent import PrognosisAgent
from c9agent.agents.literature_agent import LiteratureAgent
from c9agent.utils.llm_client import run_llm


class CentralOrchestrator:
    """
    中央调度器 —— 三层架构的核心。

    使用方式:
        orch = CentralOrchestrator()
        orch.register_agent(PhenotypeAgent())
        orch.register_agent(PrognosisAgent())

        report = orch.analyze_single_patient(
            ALSPatientInput(patient_id="P001", free_text="64岁男性，球部起病ALS...")
        )
        print(report)  # 完整的 AnalysisReport
    """

    def __init__(self, enable_reflection: bool = True):
        self.memory = MemoryBank()
        self.agents: dict[str, BaseAgent] = {}
        self.reflection = SelfReflectionLoop(max_iterations=3)
        self.report_builder = TraceableReportBuilder()
        self.enable_reflection = enable_reflection
        self._register_default_agents()
        print("[Orchestrator] 初始化完成，已注册 Agent:",
              list(self.agents.keys()))
        print(f"[Orchestrator] 自我反思: {'启用' if enable_reflection else '关闭'}")

    def _register_default_agents(self):
        """注册 Phase 1+2 的所有 Agent"""
        for agent in [
            PhenotypeAgent(),
            PrognosisAgent(),
            LiteratureAgent(use_mock=True),  # Phase 2: 文献检索
        ]:
            self.register_agent(agent)

    def register_agent(self, agent: BaseAgent):
        """注册一个 Agent 到调度器"""
        agent.bind_memory(self.memory)
        self.agents[agent.name] = agent

    # ========================================================================
    # 单患者分析 —— 主入口
    # ========================================================================

    def analyze_single_patient(
        self,
        input_data: ALSPatientInput | ALSPatientData,
        verbose: bool = True,
    ) -> dict:
        """
        对单个 ALS 患者执行完整分析。

        参数:
            input_data: ALSPatientInput（原始输入）或 ALSPatientData（已规范化）
            verbose: 是否打印进度

        返回:
            包含完整分析报告的字典
        """
        t_start = datetime.now()
        self._log(verbose, "=" * 60)
        self._log(verbose, "ALS 智能分析开始")
        self._log(verbose, "=" * 60)

        # —— Step 1: 输入标准化 ——
        self._log(verbose, "\n[Step 1/4] 输入标准化...")
        patient = self._normalize_input(input_data)
        self.memory.store("patient", "context", {
            "patient_id": patient.patient_id,
            "age": patient.age_at_onset,
            "onset": patient.onset_site.value,
            "alsfrsr": patient.latest_alsfrsr.total_score if patient.latest_alsfrsr else None,
            "slope": patient.alsfrsr_slope,
        }, source="Orchestrator")
        self._log(verbose, f"  患者: {patient.patient_id}, "
                  f"发病年龄={patient.age_at_onset}岁, "
                  f"起病={patient.onset_site.value}")

        # —— Step 2: 并行 Agent 调度 ——
        self._log(verbose, "\n[Step 2/4] Agent 分析中...")
        agent_results = self._dispatch_agents(patient)
        for name, result in agent_results.items():
            self.memory.store("agent_results", name, result.data,
                            source=name)
            status_icon = "[OK]" if result.status == "success" else "[WARN]"
            self._log(verbose, f"  {status_icon} {name}: {result.status} "
                      f"(confidence={result.confidence:.2f})")
            for w in result.warnings:
                self._log(verbose, f"    [!] {w}")

        # —— Step 3: 证据合成 + 自我反思 ——
        self._log(verbose, "\n[Step 3/4] 证据合成 + 自我反思...")
        synthesis = self._synthesize_evidence(patient, agent_results)

        # 反思循环
        reflection_result = None
        if self.enable_reflection:
            reflection_result = self.reflection.run(
                memory=self.memory,
                patient_summary=self._patient_to_text(patient),
                agent_outputs=str(agent_results),
            )
            self.memory.add_reflection(
                f"反思完成: {reflection_result.total_iterations} 轮迭代, "
                f"{len(reflection_result.gaps_identified)} 个缺口"
            )
            self._log(verbose, f"  反思: {reflection_result.total_iterations} 轮, "
                      f"{len(reflection_result.final_hypotheses)} 个最终假设")

        self.memory.store("synthesis", "llm_synthesis",
                         synthesis, source="CentralHost")
        self._log(verbose, f"  {synthesis[:200]}...")

        # —— Step 4: 报告生成 ——
        self._log(verbose, "\n[Step 4/4] 生成报告...")
        report = self.report_builder.build(
            patient=patient,
            agent_results=agent_results,
            reflection_result=reflection_result,
            memory=self.memory,
        )

        elapsed = (datetime.now() - t_start).total_seconds()
        self._log(verbose, f"\n分析完成，耗时 {elapsed:.1f} 秒")
        report["meta"] = {
            "generated_at": t_start.isoformat(),
            "elapsed_seconds": elapsed,
            "memory_summary": self.memory.summary(),
        }
        return report

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _normalize_input(self, input_data) -> ALSPatientData:
        """规范化输入 → ALSPatientData"""
        if isinstance(input_data, ALSPatientData):
            return input_data

        if isinstance(input_data, ALSPatientInput):
            if input_data.structured_data:
                return input_data.structured_data

            # 使用 PhenotypeAgent 解析
            pheno = self.agents.get("PhenotypeAgent")
            if pheno:
                result = pheno.execute(raw_input=input_data)
                if result.status != "failed":
                    patient_dict = result.data.get("patient", {})
                    if patient_dict:
                        return ALSPatientData(**patient_dict)

            # Fallback: 最小患者
            return ALSPatientData(
                patient_id=input_data.patient_id,
                sex="male", age_at_onset=55,
                onset_site="unknown",
                diagnosis_date=datetime.now().date(),
            )

        raise TypeError(f"不支持的输入类型: {type(input_data)}")

    def _dispatch_agents(self, patient: ALSPatientData) -> dict[str, AgentResult]:
        """
        调度所有 Tier 2 Agent 执行分析。

        Phase 1: 串行执行
        Phase 3: 改为异步并行，参考 DeepRare 的并行分发模式:
            "The process begins with two parallel steps: one focusing on
             phenotype inputs and the other on genotype data."
        """
        results = {}

        for name, agent in self.agents.items():
            if name == "PhenotypeAgent":
                continue  # 已在 _normalize_input 中调用

            try:
                result = agent.execute(patient)
                results[name] = result
            except Exception as e:
                results[name] = AgentResult.failed(name, str(e))
                self._log(True, f"  ✗ {name} 执行失败: {e}")

        return results

    def _synthesize_evidence(self, patient: ALSPatientData,
                             agent_results: dict[str, AgentResult]) -> str:
        """
        LLM 证据合成 —— 这是 DeepRare 中 Central Host 的核心功能。

        将所有 Agent 输出综合成连贯的临床叙述。
        """
        # 收集 Agent 输出
        findings = []
        for name, result in agent_results.items():
            findings.append(f"\n### {name}")
            findings.append(f"状态: {result.status}, 置信度: {result.confidence}")
            if result.data:
                findings.append(str(result.data)[:500])

        prompt = f"""你是一个 ALS（肌萎缩侧索硬化）临床专家。基于以下患者数据和系统分析结果，
请综合一个简短的分析摘要（不超过300字）。

## 患者信息
- ID: {patient.patient_id}
- 性别: {patient.sex.value}
- 发病年龄: {patient.age_at_onset}岁
- 起病部位: {patient.onset_site.value}
- 最新 ALSFRS-R: {patient.latest_alsfrsr.total_score if patient.latest_alsfrsr else '未知'}/48
- 下降速率: {f'{patient.alsfrsr_slope:.2f}分/月' if patient.alsfrsr_slope else '未知'}
- King's 分期: {patient.kings_stage or '未知'}
- 携带致病变异: {'是' if patient.has_pathogenic_variant else '否'}

## 系统分析结果
{chr(10).join(findings)}

请综合以上信息，用中文给出一个简洁的临床分析摘要。"""
        return run_llm(prompt)

    def _patient_to_text(self, patient: ALSPatientData) -> str:
        """患者数据 → LLM 可读的文本摘要"""
        parts = [
            f"患者ID: {patient.patient_id}",
            f"性别: {patient.sex.value}, 发病年龄: {patient.age_at_onset}岁",
            f"起病部位: {patient.onset_site.value}",
        ]
        if patient.latest_alsfrsr:
            parts.append(f"最新ALSFRS-R: {patient.latest_alsfrsr.total_score}/48")
        if patient.alsfrsr_slope:
            parts.append(f"ΔFS: {patient.alsfrsr_slope:.2f}分/月 ({patient.progression_category})")
        parts.append(f"King's分期: {patient.kings_stage}")
        if patient.genetic_variants:
            genes = ", ".join(v.gene for v in patient.genetic_variants)
            parts.append(f"基因变异: {genes}")
        if patient.respiratory and patient.respiratory.fvc_percent_predicted:
            parts.append(f"FVC: {patient.respiratory.fvc_percent_predicted}%")
        return "\n".join(parts)

    def _collect_warnings(self, agent_results: dict[str, AgentResult]) -> list[str]:
        """收集所有 Agent 的警告"""
        all_warnings = []
        for result in agent_results.values():
            all_warnings.extend(result.warnings)
        return all_warnings

    def _log(self, verbose: bool, msg: str):
        if verbose:
            print(msg)
