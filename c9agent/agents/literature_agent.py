"""
c9agent/agents/literature_agent.py — 文献检索 Agent

对应 DeepRare 论文的 Knowledge Searcher + Case Searcher 组件。
职责:
1. 根据患者特征自动生成 PubMed 检索策略
2. 检索文献 → 排序 → 提取证据语句
3. LLM 辅助总结关键发现
4. 返回结构化的文献证据表（用于可追溯报告）

工作流:
  患者特征 → 生成检索式 → PubMed API → 排序 → 提取证据 → LLM 总结 → 输出
"""

import json
import time
from c9agent.data.patient_schema import ALSPatientData
from c9agent.data.clinical_terms import (
    generate_search_queries, ALS_GENE_INFO, ALS_MEDICATIONS,
)
from c9agent.data.external_sources import (
    PubMedSearcher, PubMedArticle, MOCK_PUBMED_ARTICLES,
)
from c9agent.agents.base_agent import BaseAgent, AgentResult
from c9agent.utils.llm_client import run_llm


class LiteratureAgent(BaseAgent):
    """
    文献检索 Agent —— 自动检索 + 证据提取 + LLM 总结。

    使用:
        agent = LiteratureAgent()
        result = agent.execute(patient)
        for article in result.data["articles"]:
            print(article["title"], article["relevance_score"])
    """

    def __init__(self, email: str = "", use_mock: bool = True):
        """
        参数:
            email: PubMed API 需要邮箱
            use_mock: True = 先用模拟数据测试（离线可用）
        """
        super().__init__(
            name="LiteratureAgent",
            description="PubMed文献检索 + 证据提取 + LLM总结",
        )
        self.use_mock = use_mock
        self.searcher = PubMedSearcher(
            email=email or "",
            cache_db="data/literature_cache.db" if not use_mock else None,
        )

    def execute(self, patient: ALSPatientData,
                max_articles: int = 10, **kwargs) -> AgentResult:
        """执行文献检索"""
        t0 = time.time()
        warnings = []

        # —— Step 1: 生成检索策略 ——
        queries = generate_search_queries(patient)
        self._add_evidence(
            "observation",
            f"为患者 {patient.patient_id} 生成 {len(queries)} 个检索主题",
            confidence=1.0,
        )

        # —— Step 2: 执行检索 ——
        all_articles = []
        for q in queries:
            try:
                if self.use_mock:
                    articles = self._mock_search(q)
                else:
                    articles = self.searcher.search(
                        q["query"],
                        max_results=max(3, max_articles // len(queries)),
                    )
                all_articles.extend(articles)
            except Exception as e:
                warnings.append(f"检索失败 [{q['topic']}]: {e}")

        # 去重
        seen_pmids = set()
        unique_articles = []
        for a in all_articles:
            if a.pmid not in seen_pmids:
                seen_pmids.add(a.pmid)
                unique_articles.append(a)

        # —— Step 3: 相关性排序 ——
        ranked = self._rank_articles(unique_articles, patient)[:max_articles]

        # —— Step 4: 提取证据语句 ——
        for article in ranked:
            article.evidence_statements = self._extract_evidence(article, patient)

        # —— Step 5: LLM 总结 ——
        if ranked and not self.use_mock:
            synthesis = self._llm_summarize(ranked, patient)
        elif ranked:
            synthesis = self._simple_summary(ranked, patient)
        else:
            synthesis = "未找到相关文献"

        # —— 证据节点 ——
        for article in ranked[:5]:
            self._add_evidence(
                "evidence",
                f"[{article.journal} {article.year}] {article.title[:120]}",
                source=f"PMID:{article.pmid}",
                confidence=0.7,
            )

        elapsed = (time.time() - t0) * 1000

        return AgentResult(
            agent_name=self.name,
            status="success" if not warnings else "partial",
            data={
                "articles": [self._article_to_dict(a) for a in ranked],
                "synthesis": synthesis,
                "search_queries": [q["query"] for q in queries[:5]],
            },
            evidence_nodes=[],
            confidence=0.70 if ranked else 0.30,
            warnings=warnings,
            execution_time_ms=elapsed,
        )

    # —— 模拟检索（离线测试用） ——

    def _mock_search(self, query: dict) -> list[PubMedArticle]:
        """用模拟数据替代 PubMed API（离线可用）"""
        topic = query.get("topic", "").lower()
        results = []
        for article in MOCK_PUBMED_ARTICLES:
            score = 0.0
            # 简单关键词匹配
            title_lower = article.title.lower()
            abstract_lower = article.abstract.lower()
            for keyword in topic.split():
                if keyword.lower() in title_lower:
                    score += 0.3
                if keyword.lower() in abstract_lower:
                    score += 0.1
            if score > 0:
                article.relevance_score = min(score, 1.0)
                results.append(article)
        return sorted(results, key=lambda a: a.relevance_score, reverse=True)

    # —— 排序 ——

    def _rank_articles(self, articles: list[PubMedArticle],
                       patient: ALSPatientData) -> list[PubMedArticle]:
        """基于患者特征对文献做相关性排序"""
        # 关键词权重
        boost_terms = []

        # 起病部位
        if patient.is_bulbar_onset:
            boost_terms.append(("bulbar", 0.2))

        # 基因
        for v in patient.genetic_variants:
            boost_terms.append((v.gene.lower(), 0.3))

        # 进展速度
        if patient.alsfrsr_slope and patient.alsfrsr_slope > 0.89:
            boost_terms.extend([("rapid progression", 0.15), ("fast progress", 0.15)])

        for article in articles:
            score = article.relevance_score
            text = (article.title + " " + article.abstract).lower()
            for term, boost in boost_terms:
                if term in text:
                    score += boost
            article.relevance_score = min(score, 1.0)

        return sorted(articles, key=lambda a: a.relevance_score, reverse=True)

    # —— 证据提取 ——

    def _extract_evidence(self, article: PubMedArticle,
                          patient: ALSPatientData) -> list[str]:
        """从摘要中提取临床证据语句（简化版：提取含数字的句子）"""
        if not article.abstract:
            return []
        sentences = [s.strip() + "." for s in article.abstract.split(". ") if s.strip()]
        evidence = []
        for s in sentences:
            # 选择含临床数据的句子
            if any(kw in s.lower() for kw in [
                "survival", "median", "HR", "hazard ratio", "hazard",
                "progression", "months", "p<", "p =", "predict",
                "associated", "significant", "bulbar", "C9orf72",
            ]):
                evidence.append(s[:200])
        return evidence[:5]  # 最多5条

    # —— LLM 总结 ——

    def _llm_summarize(self, articles: list[PubMedArticle],
                        patient: ALSPatientData) -> str:
        """用 LLM 生成文献证据综合摘要"""
        articles_text = "\n".join(
            f"[{a.pmid}] {a.title}\n{a.abstract[:300]}\n"
            for a in articles[:5]
        )

        prompt = f"""你是一个 ALS 临床研究专家。基于以下文献和患者信息，用中文给出一个简短的文献证据综合。

患者: {patient.sex.value}, {patient.age_at_onset}岁, {patient.onset_site.value}起病

文献摘要:
{articles_text[:3000]}

请用3-5句话总结这些文献对理解该患者预后的关键启示。"""
        return run_llm(prompt)

    def _simple_summary(self, articles: list[PubMedArticle],
                        patient: ALSPatientData) -> str:
        """不依赖 LLM 的文献总结（离线模式）"""
        if not articles:
            return "未找到与该患者特征相关的文献。"

        key_findings = []
        for a in articles[:3]:
            if a.evidence_statements:
                key_findings.append(f"[{a.journal} {a.year}] {a.evidence_statements[0]}")

        return (
            f"检索到 {len(articles)} 篇相关文献。"
            f"关键发现: {'; '.join(key_findings[:3])}"
            if key_findings
            else f"检索到 {len(articles)} 篇相关文献，详细摘要请查看原文。"
        )

    def _article_to_dict(self, article: PubMedArticle) -> dict:
        return {
            "pmid": article.pmid,
            "title": article.title,
            "abstract": article.abstract[:500],
            "authors": article.authors[:5],
            "journal": article.journal,
            "year": article.year,
            "url": article.pubmed_url,
            "relevance_score": round(article.relevance_score, 2),
            "evidence_statements": article.evidence_statements,
        }
