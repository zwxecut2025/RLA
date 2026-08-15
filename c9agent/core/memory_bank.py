"""
c9agent/core/memory_bank.py — 上下文记忆银行

这是 DeepRare 论文中 Central Host 的 Memory Bank 组件的实现。

作用:
    在工作流执行期间累积所有中间结果 —— Agent 输出、检索到的文献、
    生成的假设、反思日志等。最终传给 ReportBuilder 生成可追溯报告。

与 DeepRare 论文的对应:
    DeepRare: "The memory bank is initialized as empty and updated incrementally
              with information gathered by agent servers."
    本模块: MemoryBank 类提供结构化槽位存储，支持按时间戳追溯。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """一条记忆记录"""
    key: str                        # 记忆的名称（如 "patient_context"）
    value: Any                      # 记忆的内容
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""                # 来源（哪个 Agent 写入的）
    note: str = ""                  # 备注


class MemoryBank:
    """
    中央记忆银行 —— 工作流的"工作台"。

    DeepRare 描述:
    "The memory bank is initialized as empty and updated incrementally
     with information gathered by agent servers."

    使用方式:
        memory = MemoryBank()
        memory.store("phenotype", "hp_001", {"hpo_term": "HP:0001260", ...})
        ...
        all_context = memory.to_context_string()  # 传给 LLM
    """

    def __init__(self):
        # 用嵌套字典组织: slot -> key -> value
        self._store: dict[str, dict[str, MemoryEntry]] = {}
        # 推理链（特殊处理，需要保持顺序）
        self._reasoning_chain: list[dict] = []
        # 假设历史
        self._hypotheses: list[dict] = []
        # 反思日志
        self._reflection_log: list[str] = []

    # —— 基本存取 ——

    def store(self, slot: str, key: str, value: Any,
              source: str = "", note: str = "") -> None:
        """
        存入一条记忆。

        参数:
            slot: 记忆槽（如 "phenotype", "genotype", "prognosis", "literature"）
            key: 记忆键（如 "hp_001", "variant_c9orf72"）
            value: 任意 Python 对象
            source: 来源 Agent 名称
            note: 备注
        """
        if slot not in self._store:
            self._store[slot] = {}
        self._store[slot][key] = MemoryEntry(
            key=key, value=value, source=source, note=note
        )

    def retrieve(self, slot: str, key: str) -> Any | None:
        """取出一条记忆"""
        entry = self._store.get(slot, {}).get(key)
        return entry.value if entry else None

    def get_slot(self, slot: str) -> dict[str, Any]:
        """获取一个槽位下的所有记忆（key -> value 字典）"""
        return {k: e.value for k, e in self._store.get(slot, {}).items()}

    # —— 推理链特殊操作 ——

    def add_reasoning_node(self, node: dict) -> None:
        """
        添加一个推理节点到推理链。

        node 格式（参考 DeepRare 的 EvidenceNode）:
        {
            "node_id": "obs_001",
            "type": "observation|inference|hypothesis|evidence|conclusion",
            "content": "...",
            "source": "PMID:12345" | None,
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345/" | None,
            "confidence": 0.92,
            "parent_ids": ["obs_001"],
        }
        """
        self._reasoning_chain.append(node)

    def get_reasoning_chain(self) -> list[dict]:
        """获取完整推理链（按添加顺序）"""
        return list(self._reasoning_chain)

    # —— 假设管理 ——

    def add_hypothesis(self, hypothesis: dict) -> None:
        """添加一个诊断/预后假设"""
        hypothesis["timestamp"] = datetime.now().isoformat()
        self._hypotheses.append(hypothesis)

    # —— 反思日志 ——

    def add_reflection(self, entry: str) -> None:
        """添加一条反思记录"""
        self._reflection_log.append(
            f"[{datetime.now().isoformat()}] {entry}"
        )

    # —— 序列化 ——

    def to_context_string(self, max_items_per_slot: int = 10) -> str:
        """
        将所有记忆序列化为 LLM 可读取的上下文字符串。

        这是 Central Host 在做证据合成时传给 LLM 的内容。
        """
        parts = []

        for slot_name, entries in self._store.items():
            parts.append(f"\n## {slot_name.upper()}")
            for i, (key, entry) in enumerate(entries.items()):
                if i >= max_items_per_slot:
                    parts.append(f"  ... (共 {len(entries)} 条，已截断)")
                    break
                parts.append(f"  [{key}] (来源: {entry.source})")
                parts.append(f"    {_format_value(entry.value)}")

        if self._reasoning_chain:
            parts.append(f"\n## 推理链 ({len(self._reasoning_chain)} 节点)")
            for node in self._reasoning_chain:
                parts.append(f"  [{node.get('type', '?')}] {node.get('content', '')[:100]}")

        if self._reflection_log:
            parts.append(f"\n## 反思日志 ({len(self._reflection_log)} 条)")
            for entry in self._reflection_log[-5:]:  # 只取最近5条
                parts.append(f"  {entry}")

        return "\n".join(parts)

    def summary(self) -> dict:
        """返回记忆库的统计摘要"""
        return {
            "total_slots": len(self._store),
            "total_entries": sum(len(e) for e in self._store.values()),
            "slots": list(self._store.keys()),
            "reasoning_nodes": len(self._reasoning_chain),
            "hypotheses": len(self._hypotheses),
            "reflections": len(self._reflection_log),
        }


def _format_value(value: Any) -> str:
    """格式化任意值为简短字符串（用于 LLM 上下文）"""
    if isinstance(value, str):
        return value[:200]
    elif isinstance(value, dict):
        return str(value)[:200]
    elif isinstance(value, list):
        return f"[{len(value)} 项]"
    else:
        return str(value)[:200]
