"""
c9agent/data/knowledge/knowledge_base.py — ALS 临床知识库

本地结构化知识存储，用于：
1. 解释预测依据（每个特征如何影响生存预测）
2. 提供临床背景知识（基因、药物、分期、指南）
3. 为报告生成可追溯的文献引用

设计原则：
- Markdown 文件存储知识内容（易编辑、人类可读）
- YAML frontmatter 提供结构化元数据
- 关键词匹配检索（无需外部依赖，离线可用）
- 对 LLM 上下文友好：只返回最相关的 3-5 个片段
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


class KnowledgeBase:
    """
    ALS 临床知识库。

    扫描 knowledge/ 目录下的 .md 文件，解析 YAML frontmatter，
    构建索引，提供关键词检索。

    使用:
        kb = KnowledgeBase()
        results = kb.search(["球部起病", "bulbar", "快速进展"])
        for topic in results:
            print(topic["title"], topic["summary"])

        # 获取患者相关的知识片段（用于报告）
        context = kb.get_context_for_patient(
            onset_site="bulbar",
            progression="fast",
            genes=[],
            fvc=47,
        )
    """

    def __init__(self, knowledge_dir: str = None):
        self._dir = Path(knowledge_dir) if knowledge_dir else Path(__file__).parent
        self._documents: dict[str, dict] = {}       # topic_id → {meta, content, sections}
        self._index: dict[str, list[str]] = {}       # keyword → [topic_id, ...]
        self._index_data: dict = {}                  # 缓存 index.json
        self._load_all()

    # ========================================================================
    # 加载与索引
    # ========================================================================

    def reload(self):
        """热重载：修改 Markdown 文件后调用此方法即可生效"""
        self._documents.clear()
        self._index.clear()
        self._load_all()

    def _load_all(self):
        """扫描目录，加载所有 .md 文件并构建索引"""
        md_files = sorted(self._dir.glob("*.md"))
        if not md_files:
            print(f"[KnowledgeBase] 警告: {self._dir} 下未找到 .md 文件")
            return

        for md_file in md_files:
            try:
                doc = self._parse_markdown(md_file)
                topic_id = doc["meta"].get("id", md_file.stem)
                self._documents[topic_id] = doc
            except Exception as e:
                print(f"[KnowledgeBase] 跳过 {md_file.name}: {e}")

        self._build_index()
        self._save_index()

        print(f"[KnowledgeBase] 已加载 {len(self._documents)} 篇知识文档, "
              f"{len(self._index)} 个索引词")

    def _parse_markdown(self, filepath: Path) -> dict:
        """解析 Markdown 文件的 YAML frontmatter 和正文"""
        text = filepath.read_text(encoding="utf-8")

        # 解析 YAML frontmatter (--- ... ---)
        meta = {}
        content_start = 0
        if text.startswith("---"):
            end = text.find("---", 3)
            if end > 0:
                frontmatter = text[3:end].strip()
                meta = self._parse_yaml_simple(frontmatter)
                content_start = end + 3

        content = text[content_start:].strip()

        # 解析章节（## 标题）
        sections = {}
        current_section = "_intro"
        current_text = []
        for line in content.split("\n"):
            if line.startswith("## "):
                if current_text:
                    sections[current_section] = "\n".join(current_text).strip()
                current_section = line[3:].strip()
                current_text = []
            else:
                current_text.append(line)
        if current_text:
            sections[current_section] = "\n".join(current_text).strip()

        # 如果正文开头没有 ## 标题，整个作为 _intro
        if "_intro" in sections and not sections["_intro"]:
            del sections["_intro"]

        return {
            "meta": meta,
            "content": content,
            "sections": sections,
            "file": str(filepath),
        }

    def _parse_yaml_simple(self, text: str) -> dict:
        """简易 YAML 解析（只支持 key: value 和 key: [list]）"""
        result = {}
        current_key = None
        current_list = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # 列表项
            if stripped.startswith("- "):
                if current_key:
                    current_list.append(stripped[2:].strip().strip('"'))
                continue

            # 保存之前的列表
            if current_key and current_list:
                result[current_key] = current_list
                current_list = []

            # key: value
            if ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value.startswith("[") and value.endswith("]"):
                    # 内联列表
                    value = [v.strip().strip('"').strip("'")
                             for v in value[1:-1].split(",") if v.strip()]
                result[key] = value
                current_key = key

        # 最后未保存的列表
        if current_key and current_list:
            result[current_key] = current_list

        return result

    def _build_index(self):
        """构建关键词倒排索引"""
        self._index.clear()
        for topic_id, doc in self._documents.items():
            meta = doc.get("meta", {})
            keywords = meta.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]
            topics = meta.get("topics", [])
            if isinstance(topics, str):
                topics = [t.strip() for t in topics.split(",")]

            all_terms = list(keywords) + list(topics)
            # 也索引标题中的词
            title = meta.get("title", "")
            all_terms.extend(title.split())

            for term in all_terms:
                term_lower = term.lower().strip()
                if term_lower not in self._index:
                    self._index[term_lower] = []
                if topic_id not in self._index[term_lower]:
                    self._index[term_lower].append(topic_id)

    def _save_index(self):
        """保存索引到 index.json（供调试和外部工具使用）"""
        index_path = self._dir / "index.json"
        index_data = {
            "updated": datetime.now().isoformat(),
            "total_documents": len(self._documents),
            "total_keywords": len(self._index),
            "documents": {
                tid: {
                    "title": doc["meta"].get("title", tid),
                    "topics": doc["meta"].get("topics", []),
                    "keywords": doc["meta"].get("keywords", []),
                    "file": Path(doc.get("file", "")).name,
                }
                for tid, doc in self._documents.items()
            },
        }
        index_path.write_text(json.dumps(index_data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        self._index_data = index_data

    # ========================================================================
    # 检索
    # ========================================================================

    def search(self, query_terms: list[str], top_k: int = 5,
               category: str = None) -> list[dict]:
        """
        关键词检索。

        参数:
            query_terms: 查询词列表（中文/英文）
            top_k: 返回结果数
            category: 可选，限定知识类别

        返回:
            [{topic_id, title, category, summary, sections, score, ...}]
        """
        # 计算每个文档的匹配分数
        scores: dict[str, float] = {}
        for term in query_terms:
            term_lower = term.lower().strip()
            # 精确匹配
            if term_lower in self._index:
                for tid in self._index[term_lower]:
                    scores[tid] = scores.get(tid, 0) + 1.0
            # 部分匹配
            else:
                for idx_term, tids in self._index.items():
                    if term_lower in idx_term or idx_term in term_lower:
                        for tid in tids:
                            scores[tid] = scores.get(tid, 0) + 0.5

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for tid, score in ranked:
            doc = self._documents.get(tid)
            if not doc:
                continue

            # 类别过滤
            if category and doc["meta"].get("category") != category:
                continue

            meta = doc["meta"]
            results.append({
                "topic_id": tid,
                "title": meta.get("title", tid),
                "category": meta.get("category", ""),
                "summary": meta.get("summary", ""),
                "sections": list(doc.get("sections", {}).keys()),
                "references": meta.get("references", []),
                "score": round(score / max(len(query_terms), 1), 3),
            })

            if len(results) >= top_k:
                break

        return results

    def get_document(self, topic_id: str) -> Optional[dict]:
        """获取完整文档"""
        return self._documents.get(topic_id)

    def get_section(self, topic_id: str, section_title: str) -> Optional[str]:
        """获取文档的指定章节"""
        doc = self._documents.get(topic_id)
        if not doc:
            return None
        return doc.get("sections", {}).get(section_title)

    def list_documents(self) -> list[dict]:
        """列出所有文档"""
        return [
            {
                "topic_id": tid,
                "title": doc["meta"].get("title", tid),
                "category": doc["meta"].get("category", ""),
            }
            for tid, doc in self._documents.items()
        ]

    # ========================================================================
    # 患者相关检索
    # ========================================================================

    def get_context_for_patient(self, onset_site: str = "unknown",
                                progression: str = "unknown",
                                genes: list[str] = None,
                                fvc: float = None,
                                age: float = None,
                                max_items: int = 8) -> list[dict]:
        """
        根据患者特征检索相关知识。

        参数:
            onset_site: 起病部位 (bulbar/spinal_cervical/spinal_lumbar)
            progression: 进展速度 (fast/moderate/slow)
            genes: 基因变异列表
            fvc: FVC 百分比
            age: 发病年龄
            max_items: 最多返回条数

        返回:
            相关知识片段列表
        """
        query_terms = ["预后因子", "生存预测"]  # 总是包含

        # 起病部位
        if onset_site == "bulbar":
            query_terms.extend(["球部起病", "bulbar", "吞咽", "构音"])
        elif onset_site in ("spinal_cervical", "spinal_lumbar"):
            query_terms.extend(["肢体起病", "spinal"])

        # 进展速度
        if progression == "fast":
            query_terms.extend(["快速进展", "rapid", "high risk"])
        elif progression == "slow":
            query_terms.extend(["缓慢进展", "slow"])

        # 基因
        for gene in (genes or []):
            query_terms.append(gene)
            if gene == "C9orf72":
                query_terms.extend(["C9orf72", "额颞叶", "FTD"])
            elif gene == "SOD1":
                query_terms.extend(["SOD1", "抗氧化"])

        # FVC
        if fvc is not None:
            query_terms.append("FVC")
            if fvc < 70:
                query_terms.extend(["呼吸衰竭", "NIV", "无创通气"])

        # 年龄
        if age is not None and age > 60:
            query_terms.append("高龄")
        elif age is not None and age < 40:
            query_terms.append("年轻起病")

        # 搜索
        all_results = self.search(query_terms, top_k=max_items * 2)

        # 确保方法论等核心文档总是包含
        core_ids = ["model_methodology", "prognostic_factors"]
        final_results = []
        seen = set()

        for tid in core_ids:
            doc = self._documents.get(tid)
            if doc:
                final_results.append({
                    "topic_id": tid,
                    "title": doc["meta"].get("title", tid),
                    "category": doc["meta"].get("category", ""),
                    "summary": doc["meta"].get("summary", ""),
                    "references": doc["meta"].get("references", []),
                })
                seen.add(tid)

        for r in all_results:
            if r["topic_id"] not in seen and len(final_results) < max_items:
                final_results.append(r)
                seen.add(r["topic_id"])

        return final_results

    def format_for_report(self, patient_context: list[dict],
                          max_chars: int = 2000) -> str:
        """
        格式化知识片段为报告可用的 Markdown 文本。

        控制长度以适应 LLM 上下文窗口（2048 tokens ≈ 2000 中文字符）。
        """
        if not patient_context:
            return ""

        lines = []
        total = 0

        for item in patient_context:
            header = f"### {item['title']} [{item.get('category', '')}]"
            summary = item.get("summary", "")
            refs = item.get("references", [])

            entry = f"{header}\n{summary}\n"
            if refs:
                ref_str = "；".join(
                    r if isinstance(r, str) else r.get("citation", "")
                    for r in refs[:2]
                )
                entry += f"> 文献: {ref_str}\n"
            entry += "\n"

            if total + len(entry) > max_chars:
                lines.append(f"> ... 共 {len(patient_context)} 条，截断至 {len(lines)} 条")
                break

            lines.append(entry)
            total += len(entry)

        return "\n".join(lines)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def keyword_count(self) -> int:
        return len(self._index)
