"""
c9agent/data/external_sources.py — 外部数据源连接器

PubMed E-utilities API (免费) + ClinVar API。
设计为可离线测试：如果 API 不可用，返回模拟数据。

DeepRare 论文中对应的 Knowledge Searcher 组件:
- General Web search (Bing, Google, DuckDuckGo)
- Medical domain search (PubMed, Orphanet, OMIM, HPO)
这里先实现最核心的 PubMed + ClinVar。
"""

import json
import time
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
import httpx


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class PubMedArticle:
    """一篇 PubMed 文献"""
    pmid: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: int = 0
    doi: str = ""
    url: str = ""
    keywords: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    evidence_statements: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        authors = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors += " et al."
        return f"{authors}. {self.title}. {self.journal}. {self.year}."

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclass
class ClinVarRecord:
    """一条 ClinVar 变异记录"""
    variation_id: str
    gene: str
    hgvs_c: str = ""
    hgvs_p: str = ""
    clinical_significance: str = ""  # Pathogenic / VUS / Benign
    review_status: str = ""          # criteria provided / expert panel
    condition: str = ""
    last_evaluated: str = ""
    url: str = ""

    @property
    def is_pathogenic(self) -> bool:
        return "pathogenic" in self.clinical_significance.lower()


# ============================================================================
# PubMed 检索器
# ============================================================================

class PubMedSearcher:
    """
    PubMed E-utilities API 封装。

    关键限制:
    - 无 API Key: 3 req/sec
    - 有 API Key: 10 req/sec
    - 返回最多 10,000 条（实际我们用 20-50）

    使用:
        searcher = PubMedSearcher(email="your@email.com")
        articles = searcher.search("C9orf72 ALS survival", max_results=10)
        for a in articles:
            print(a.pmid, a.title)
    """

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str = "", api_key: str = "",
                 cache_db: str = None):
        """
        参数:
            email: NCBI 要求提供邮箱（使用条款）
            api_key: 提高速率限制（可选）
            cache_db: SQLite 缓存路径（避免重复请求）
        """
        self.email = email or "user@example.com"
        self.api_key = api_key
        self._last_request = 0.0
        self._min_interval = 1.0 / 3  # 3 req/sec (无 key)
        if api_key:
            self._min_interval = 1.0 / 10

        # SQLite 缓存
        self._cache_db = cache_db
        if cache_db:
            self._init_cache()

    def _rate_limit(self):
        """API 限速"""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _get(self, params: dict) -> dict:
        """发送 GET 请求到 E-utilities"""
        params.setdefault("email", self.email)
        if self.api_key:
            params["api_key"] = self.api_key

        self._rate_limit()
        with httpx.Client(timeout=30) as client:
            resp = client.get(self.BASE + "/esearch.fcgi", params=params)
            resp.raise_for_status()
            return self._parse_xml(resp.text)

    def _parse_xml(self, text: str) -> dict:
        """简单 XML 解析（避免额外依赖）"""
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
        result = {}
        for child in root:
            result[child.tag] = child.text or ""
            # 处理列表子元素
            sub_list = []
            for sub in child:
                sub_list.append(sub.text or "")
            if sub_list:
                result[child.tag + "_list"] = sub_list
        return result

    # —— 核心搜索 ——

    def search(self, query: str, max_results: int = 20,
               years_back: int = 10) -> list[PubMedArticle]:
        """
        搜索 PubMed 并返回带摘要的文章列表。

        参数:
            query: PubMed 查询字符串（支持 MeSH 标签）
            max_results: 最大返回数量
            years_back: 只返回最近 N 年的文章
        """
        # 检查缓存
        cache_key = self._cache_key(query, max_results, years_back)
        if self._cache_db:
            cached = self._cache_get(cache_key)
            if cached:
                return cached

        # Step 1: esearch — 获取 PMID 列表
        search_result = self._get({
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "xml",
            "sort": "relevance",
            "mindate": f"{datetime.now().year - years_back}/01/01",
            "maxdate": f"{datetime.now().year}/12/31",
            "datetype": "pdat",
        })

        id_list = search_result.get("IdList_list", [])
        if not id_list:
            return []

        # Step 2: efetch — 获取摘要
        abstracts = self._fetch_abstracts(id_list)

        # 缓存
        if self._cache_db:
            self._cache_set(cache_key, abstracts)

        return abstracts

    def _fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """批量获取摘要"""
        self._rate_limit()
        params = {
            "db": "pubmed",
            "id": ",".join(pmids[:50]),  # 最多一次取50篇
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        with httpx.Client(timeout=30) as client:
            resp = client.get(self.BASE + "/efetch.fcgi", params=params)
            resp.raise_for_status()
            return self._parse_articles(resp.text)

    def _parse_articles(self, xml_text: str) -> list[PubMedArticle]:
        """解析 efetch 返回的 XML → PubMedArticle 列表"""
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        articles = []

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                medline = article_elem.find(".//MedlineCitation")
                article = medline.find(".//Article") if medline is not None else None
                if article is None:
                    continue

                # PMID
                pmid_elem = medline.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""

                # 标题
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text or "" if title_elem is not None else ""

                # 摘要
                abstract_parts = []
                abstract_elem = article.find(".//Abstract")
                if abstract_elem is not None:
                    for part in abstract_elem.findall(".//AbstractText"):
                        label = part.get("Label", "")
                        text = part.text or ""
                        abstract_parts.append(f"{label}: {text}" if label else text)
                abstract = " ".join(abstract_parts)

                # 作者
                authors = []
                for author in article.findall(".//Author"):
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {fore}")

                # 期刊
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""

                # 年份
                year_elem = article.find(".//PubDate/Year")
                year = int(year_elem.text) if year_elem is not None and year_elem.text else 0

                # DOI
                doi = ""
                for eid in article.findall(".//ELocationID"):
                    if eid.get("EIdType") == "doi":
                        doi = eid.text or ""

                articles.append(PubMedArticle(
                    pmid=pmid,
                    title=title,
                    abstract=abstract[:2000],  # 截断过长摘要
                    authors=authors,
                    journal=journal,
                    year=year,
                    doi=doi,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                ))
            except Exception:
                continue

        return articles

    # —— 缓存 ——

    def _init_cache(self):
        Path(self._cache_db).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pubmed_cache (
                    cache_key TEXT PRIMARY KEY,
                    articles_json TEXT,
                    created_at TEXT
                )
            """)

    def _cache_key(self, query: str, n: int, years: int) -> str:
        return hashlib.md5(f"{query}|{n}|{years}".encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[list[PubMedArticle]]:
        try:
            with sqlite3.connect(self._cache_db) as conn:
                row = conn.execute(
                    "SELECT articles_json FROM pubmed_cache WHERE cache_key=?",
                    (key,)
                ).fetchone()
                if row:
                    data = json.loads(row[0])
                    return [PubMedArticle(**a) for a in data]
        except Exception:
            pass
        return None

    def _cache_set(self, key: str, articles: list[PubMedArticle]):
        try:
            with sqlite3.connect(self._cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO pubmed_cache VALUES (?, ?, ?)",
                    (key, json.dumps([a.__dict__ for a in articles], default=str),
                     datetime.now().isoformat())
                )
        except Exception:
            pass


# ============================================================================
# ClinVar 查询器
# ============================================================================

class ClinVarSearcher:
    """
    ClinVar E-utilities API 封装。

    使用:
        searcher = ClinVarSearcher()
        record = searcher.query_variant("SOD1", "NM_000454.5:c.14C>T")
    """

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def query_variant(self, gene: str, hgvs: str = "",
                      variation_id: str = "") -> Optional[ClinVarRecord]:
        """查询 ClinVar 变异记录"""
        if variation_id:
            query = f"{variation_id}[Variant ID]"
        elif hgvs:
            query = f"{gene}[Gene] AND {hgvs}"
        else:
            query = f"{gene}[Gene] AND ALS"

        params = {
            "db": "clinvar",
            "term": query,
            "retmax": "1",
            "retmode": "xml",
        }

        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(self.BASE + "/esearch.fcgi", params=params)
                resp.raise_for_status()

                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.text)
                id_list = [e.text for e in root.findall(".//Id")]

                if not id_list:
                    return None

                # efetch 获取详细信息
                detail_params = {
                    "db": "clinvar",
                    "id": id_list[0],
                    "retmode": "xml",
                }
                resp2 = client.get(self.BASE + "/efetch.fcgi", params=detail_params)
                root2 = ET.fromstring(resp2.text)

                vc = root2.find(".//VariationClinicalSignificance")
                significance = ""
                if vc is not None:
                    desc = vc.find(".//Description")
                    significance = desc.text if desc is not None else ""

                review = root2.find(".//ReviewStatus")
                review_status = review.text if review is not None else ""

                condition_elem = root2.find(".//TraitSet//Name/ElementValue")
                condition = condition_elem.text if condition_elem is not None else ""

                return ClinVarRecord(
                    variation_id=id_list[0],
                    gene=gene,
                    hgvs_c=hgvs,
                    clinical_significance=significance,
                    review_status=review_status,
                    condition=condition,
                    url=f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{id_list[0]}/",
                )
        except Exception as e:
            print(f"[ClinVar] 查询失败: {e}")
            return None


# ============================================================================
# 离线测试用模拟数据
# ============================================================================

MOCK_PUBMED_ARTICLES = [
    PubMedArticle(
        pmid="29598923",
        title="Prognosis for patients with amyotrophic lateral sclerosis: "
              "development and validation of a personalised prediction model",
        abstract="Background: Amyotrophic lateral sclerosis (ALS) is a heterogeneous "
                 "disease. We developed and validated a model for predicting survival "
                 "without non-invasive ventilation (>23 h/day), tracheostomy, or death. "
                 "Methods: Data from 11,475 patients across 14 European centres (1992-2016). "
                 "Eight predictors were selected: bulbar onset (HR 1.71), age at onset "
                 "(HR 1.03/year), definite vs probable/possible ALS (HR 1.47), diagnostic "
                 "delay (HR 0.52), FVC (HR 0.988/%), progression rate (HR 6.33), FTD "
                 "(HR 1.34), C9orf72 (HR 1.45). External validation c-statistic 0.78.",
        authors=["Westeneng HJ", "Debray TPA", "Visser AE"],
        journal="Lancet Neurology", year=2018,
        url="https://pubmed.ncbi.nlm.nih.gov/29598923/",
        relevance_score=0.98,
    ),
    PubMedArticle(
        pmid="34864363",
        title="Predictors of survival in patients with amyotrophic lateral sclerosis: "
              "A large meta-analysis",
        abstract="Background: The prognostic factors for ALS have been extensively studied. "
                 "This meta-analysis included 115 studies with 55,169 patients. The strongest "
                 "predictors of poor prognosis were CSF NfL (HR 6.80), FTD (HR 2.98), "
                 "ALSFRS-R decline ≥1pt/month (HR 2.37), and respiratory onset (HR 2.20). "
                 "Protective factors included pure UMN/LMN phenotype (HR 0.32), diagnostic "
                 "delay ≥12 months (HR 0.38), higher baseline ALSFRS-R (HR 0.95/point), "
                 "higher FVC (HR 0.98/%). Gender showed no significant association.",
        authors=["Su WM", "Cheng YF", "Jiang Z", "Duan QQ", "Yang TM", "Shang HF", "Chen YP"],
        journal="EBioMedicine", year=2021,
        url="https://pubmed.ncbi.nlm.nih.gov/34864363/",
        relevance_score=0.95,
    ),
    PubMedArticle(
        pmid="16434671",
        title="Progression rate of ALSFRS-R at time of diagnosis predicts "
              "survival time in ALS",
        abstract="Objective: To determine whether the rate of ALSFRS-R decline at "
                 "diagnosis predicts survival. Methods: 82 ALS patients prospectively "
                 "followed. The mean ALSFRS-R slope was 0.89 points/month. Patients "
                 "with slope >0.89 had significantly shorter survival than those with "
                 "slope <0.89 (p<0.001). The progression rate at diagnosis is a strong "
                 "and independent predictor of survival in ALS.",
        authors=["Kimura F", "Fujimura C", "Ishida S"],
        journal="Neurology", year=2006,
        url="https://pubmed.ncbi.nlm.nih.gov/16434671/",
        relevance_score=0.90,
    ),
]
