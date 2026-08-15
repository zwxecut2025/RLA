"""
c9agent/agents/base_agent.py — Agent 基类和 I/O 协议

参考 DeepRare 论文的 Agent Server 设计:
每个 Agent Server 管理一个或多个专用工具，与外部环境交互，
通过标准化的 AgentRequest → AgentResult 协议与 Central Host 通信。

设计要点:
1. 所有 Tier 2 Agent 继承 BaseAgent，实现 execute() 方法
2. 输入永远是 ALSPatientData（已被 PhenotypeAgent 规范化）
3. 输出永远是 AgentResult（统一的返回格式）
4. Agent 可以调用 LLM 做推理，也可以仅运行纯计算逻辑
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field

from c9agent.data.patient_schema import ALSPatientData
from c9agent.core.memory_bank import MemoryBank


# ============================================================================
# 标准化 I/O 协议
# ============================================================================

class AgentResult(BaseModel):
    """
    所有 Agent 的统一输出格式。

    参考 DeepRare:
    "Agent servers execute specialized tasks such as phenotype extraction
     and knowledge retrieval, enabling dynamic interaction with external
     data sources."

    每个 Agent 返回的数据放在 data 字段中（任意 JSON），
    evidence_nodes 包含推理链节点，用于生成可追溯报告。
    """
    agent_name: str                                    # Agent 名称
    status: str = Field(default="success",
                         pattern="^(success|partial|failed)$")
    data: dict[str, Any] = {}                          # Agent 的输出数据
    evidence_nodes: list[dict] = []                    # 推理证据节点
    confidence: float = Field(default=0.0, ge=0, le=1) # 置信度
    warnings: list[str] = []                           # 警告（数据缺失等）
    execution_time_ms: float = 0.0                     # 耗时

    @classmethod
    def success(cls, name: str, data: dict,
                evidence: list[dict] = None,
                confidence: float = 0.5,
                execution_ms: float = 0) -> "AgentResult":
        return cls(
            agent_name=name,
            status="success",
            data=data,
            evidence_nodes=evidence or [],
            confidence=confidence,
            execution_time_ms=execution_ms,
        )

    @classmethod
    def partial(cls, name: str, data: dict,
                warnings: list[str],
                confidence: float = 0.3) -> "AgentResult":
        return cls(
            agent_name=name,
            status="partial",
            data=data,
            warnings=warnings,
            confidence=confidence,
        )

    @classmethod
    def failed(cls, name: str, reason: str) -> "AgentResult":
        return cls(
            agent_name=name,
            status="failed",
            data={"error": reason},
            warnings=[reason],
            confidence=0.0,
        )


# ============================================================================
# BaseAgent 抽象类
# ============================================================================

class BaseAgent(ABC):
    """
    所有 Tier 2 专业 Agent 的基类。

    子类只需要实现 execute() 方法。其他功能（绑定 Memory、日志、
    计时）由基类统一提供。

    使用方式:
        class MyAgent(BaseAgent):
            async def execute(self, patient, **kwargs):
                # 1. 分析患者数据
                # 2. 调用外部 API / 运行模型
                # 3. 构造推理证据节点
                # 4. 写入 MemoryBank
                return AgentResult.success(name=self.name, data=...)
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.memory: Optional[MemoryBank] = None   # 由 Orchestrator 注入

    def bind_memory(self, memory: MemoryBank) -> None:
        """绑定 MemoryBank（由 Orchestrator 在工作流开始时调用）"""
        self.memory = memory

    @abstractmethod
    def execute(self, patient: ALSPatientData, **kwargs) -> AgentResult:
        """
        执行 Agent 的核心分析逻辑。

        参数:
            patient: 规范化的 ALS 患者数据
            **kwargs: Agent 特定的额外参数

        返回:
            AgentResult 包含分析结果和推理证据
        """
        ...

    def _add_evidence(self, node_type: str, content: str,
                      source: str = None, confidence: float = 0.5) -> dict:
        """快捷方法：构造一个证据节点并写入 MemoryBank（如果绑定了）"""
        node = {
            "node_id": f"{self.name}_{node_type}_{0 if not self.memory else len(self.memory.get_reasoning_chain())}",
            "type": node_type,
            "content": content,
            "source": source,
            "confidence": confidence,
            "parent_ids": [],
        }
        if self.memory:
            self.memory.add_reasoning_node(node)
        return node

    def _warn(self, msg: str) -> None:
        """快捷方法：记录到 MemoryBank（如果绑定了）"""
        if self.memory:
            self.memory.store("warnings", f"{self.name}_{datetime.now()}", msg,
                            source=self.name)

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}'>"
