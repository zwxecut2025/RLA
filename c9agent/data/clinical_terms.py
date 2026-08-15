"""
c9agent/data/clinical_terms.py — ALS 临床术语字典

中英文 ALS 术语对照 + HPO 编码 + 文献检索关键词。
LiteratureAgent 和 PhenotypeAgent 共用这个字典。
"""

# ============================================================================
# ALS 核心 HPO 表型映射
# ============================================================================

ALS_HPO_MAP = {
    # 运动神经元病核心表型
    "肌萎缩侧索硬化": {
        "en": "Amyotrophic Lateral Sclerosis",
        "hpo": "HP:0007354",
        "keywords": ["ALS", "motor neuron disease", "MND"],
    },
    "进行性肌萎缩": {
        "en": "Progressive Muscular Atrophy",
        "hpo": "HP:0003674",
        "keywords": ["PMA", "progressive muscular atrophy"],
    },

    # 起病症状
    "球部起病": {
        "en": "Bulbar onset",
        "hpo": "HP:0002500",
        "keywords": ["bulbar", "dysarthria", "dysphagia"],
    },
    "构音障碍": {
        "en": "Dysarthria",
        "hpo": "HP:0001260",
        "keywords": ["dysarthria", "speech impairment"],
    },
    "吞咽困难": {
        "en": "Dysphagia",
        "hpo": "HP:0002015",
        "keywords": ["dysphagia", "swallowing difficulty"],
    },
    "肌无力": {
        "en": "Muscle weakness",
        "hpo": "HP:0001324",
        "keywords": ["weakness", "paresis", "motor deficit"],
    },
    "肌萎缩": {
        "en": "Muscular atrophy",
        "hpo": "HP:0003202",
        "keywords": ["atrophy", "muscle wasting", "amyotrophy"],
    },
    "肌束颤动": {
        "en": "Fasciculations",
        "hpo": "HP:0002380",
        "keywords": ["fasciculation", "twitching"],
    },
    "反射亢进": {
        "en": "Hyperreflexia",
        "hpo": "HP:0001347",
        "keywords": ["hyperreflexia", "brisk reflexes", "UMN signs"],
    },
    "巴氏征阳性": {
        "en": "Babinski sign",
        "hpo": "HP:0003487",
        "keywords": ["Babinski", "extensor plantar"],
    },

    # 呼吸
    "呼吸困难": {
        "en": "Dyspnea",
        "hpo": "HP:0002094",
        "keywords": ["dyspnea", "breathlessness", "respiratory"],
    },
    "呼吸功能不全": {
        "en": "Respiratory insufficiency",
        "hpo": "HP:0002093",
        "keywords": ["respiratory failure", "ventilatory insufficiency", "NIV"],
    },
    "FVC下降": {
        "en": "Decreased FVC",
        "hpo": "HP:0005952",
        "keywords": ["FVC", "forced vital capacity", "pulmonary function"],
    },

    # 其他常见症状
    "体重下降": {
        "en": "Weight loss",
        "hpo": "HP:0001824",
        "keywords": ["weight loss", "malnutrition", "cachexia"],
    },
    "假性球麻痹": {
        "en": "Pseudobulbar affect",
        "hpo": "HP:0002193",
        "keywords": ["pseudobulbar", "emotional lability", "pathological laughing"],
    },
    "额颞叶痴呆": {
        "en": "Frontotemporal dementia",
        "hpo": "HP:0002145",
        "keywords": ["FTD", "frontotemporal", "cognitive impairment", "ALS-FTD"],
    },
}


# ============================================================================
# ALS 基因列表（带文献检索关键词）
# ============================================================================

ALS_GENE_INFO = {
    "C9orf72": {
        "zh": "C9orf72 基因",
        "type": "repeat_expansion",
        "inheritance": "autosomal_dominant",
        "frequency_in_fALS": "30-40%",
        "frequency_in_sALS": "5-7%",
        "phenotype": "球部起病多见，常伴额颞叶痴呆，进展快",
        "pubmed_queries": [
            "C9orf72 ALS phenotype",
            "C9orf72 ALS survival prognosis",
            "C9orf72 ALS-FTD",
        ],
    },
    "SOD1": {
        "zh": "SOD1 基因",
        "type": "missense",
        "inheritance": "autosomal_dominant",
        "frequency_in_fALS": "20%",
        "frequency_in_sALS": "1-2%",
        "phenotype": "下肢起病多见，进展变异大，A4V变异进展极快",
        "pubmed_queries": [
            "SOD1 ALS genotype phenotype correlation",
            "SOD1 ALS progression rate",
            "SOD1 tofersen antisense oligonucleotide",
        ],
    },
    "TARDBP": {
        "zh": "TARDBP 基因 (TDP-43)",
        "type": "missense",
        "inheritance": "autosomal_dominant",
        "frequency_in_fALS": "3-5%",
        "frequency_in_sALS": "1%",
        "phenotype": "上臂起病多见，进展中等",
        "pubmed_queries": [
            "TARDBP TDP-43 ALS phenotype",
            "TDP-43 proteinopathy ALS",
        ],
    },
    "FUS": {
        "zh": "FUS 基因",
        "type": "missense",
        "inheritance": "autosomal_dominant",
        "frequency_in_fALS": "3-5%",
        "frequency_in_sALS": "<1%",
        "phenotype": "青少年起病多见，进展快，亚洲人群中较常见",
        "pubmed_queries": [
            "FUS ALS juvenile onset",
            "FUS ALS Asian population",
            "FUS ALS genotype phenotype",
        ],
    },
    "TBK1": {
        "zh": "TBK1 基因",
        "type": "loss_of_function",
        "inheritance": "autosomal_dominant",
        "frequency_in_fALS": "1-3%",
        "phenotype": "可伴额颞叶痴呆",
        "pubmed_queries": [
            "TBK1 ALS phenotype",
            "TBK1 ALS-FTD spectrum",
        ],
    },
}


# ============================================================================
# 临床分期体系
# ============================================================================

STAGING_SYSTEMS = {
    "kings": {
        "name": "King's Clinical Staging",
        "reference": "Roche et al., 2012, Brain",
        "stages": {
            1: "一个区域受累（症状 onset）",
            2: "第二个区域受累（A: 确诊时, B: 随访中）",
            3: "第三个区域受累",
            4: "需要胃造瘘（4A）或 NIV（4B）",
        },
    },
    "mitos": {
        "name": "Milano-Torino Staging (MITOS)",
        "reference": "Chiò et al., 2015, Lancet Neurology",
        "stages": {
            0: "无功能域丧失",
            1: "一个域丧失",
            2: "两个域丧失",
            3: "三个域丧失",
            4: "四个域丧失",
            5: "死亡",
        },
    },
}

# ============================================================================
# 用药信息（中文 + PubMed 检索词）
# ============================================================================

ALS_MEDICATIONS = {
    "Riluzole": {
        "zh": "利鲁唑",
        "mechanism": "谷氨酸释放抑制剂",
        "effect": "延长生存期约 2-3 个月",
        "pubmed": "riluzole ALS survival benefit",
    },
    "Edaravone": {
        "zh": "依达拉奉",
        "mechanism": "自由基清除剂",
        "effect": "减缓早期 ALS 功能下降（约33%）",
        "pubmed": "edaravone ALS efficacy clinical trial",
    },
    "Tofersen": {
        "zh": "托弗森",
        "mechanism": "SOD1 ASO 反义寡核苷酸",
        "effect": "降低 SOD1 突变患者的 NFL 水平",
        "pubmed": "tofersen SOD1 ALS antisense oligonucleotide",
    },
    "Sodium Phenylbutyrate/Taurursodiol": {
        "zh": "苯丁酸钠/牛磺熊去氧胆酸",
        "mechanism": "内质网应激调节剂",
        "effect": "CENTAUR 试验显示减缓进展",
        "pubmed": "AMX0035 CENTAUR ALS sodium phenylbutyrate",
    },
}


# ============================================================================
# 文献搜索策略 — 按患者特征生成 PubMed 查询
# ============================================================================

def generate_search_queries(patient) -> list[dict]:
    """
    根据患者特征自动生成文献检索策略。

    参数:
        patient: ALSPatientData 实例

    返回:
        [{"topic": "主题", "query": "PubMed查询字符串", "priority": 1-5}]
    """
    queries = []

    # 基础 ALS 预后查询
    queries.append({
        "topic": "ALS 预后因素",
        "query": ('"amyotrophic lateral sclerosis"[MeSH] AND '
                  '(prognosis OR survival OR progression) AND '
                  '"humans"[Filter]'),
        "priority": 3,
    })

    # 起病部位相关
    if patient.is_bulbar_onset:
        queries.append({
            "topic": "球部起病预后",
            "query": ('"bulbar onset" AND "amyotrophic lateral sclerosis" '
                      'AND (survival OR prognosis OR progression rate)'),
            "priority": 1,
        })

    # 进展速度
    if patient.alsfrsr_slope and patient.alsfrsr_slope > 0.89:
        queries.append({
            "topic": "快速进展 ALS",
            "query": ('"rapid progression" OR "fast progression" '
                      'AND "amyotrophic lateral sclerosis" '
                      'AND "prognostic factors"'),
            "priority": 1,
        })

    # 基因型相关 — 最重要
    for variant in patient.genetic_variants:
        gene_info = ALS_GENE_INFO.get(variant.gene, {})
        if gene_info and "pubmed_queries" in gene_info:
            for q in gene_info["pubmed_queries"]:
                queries.append({
                    "topic": f"{variant.gene} 基因型-表型",
                    "query": q,
                    "priority": 1,
                })

    # 呼吸功能
    if patient.respiratory and patient.respiratory.fvc_percent_predicted:
        fvc = patient.respiratory.fvc_percent_predicted
        if fvc < 70:
            queries.append({
                "topic": "ALS 呼吸衰竭预测",
                "query": ('"non-invasive ventilation" OR "respiratory failure" '
                          'AND "amyotrophic lateral sclerosis" '
                          'AND (timing OR prediction OR survival)'),
                "priority": 2,
            })

    # 用药相关
    for med in patient.medications:
        drug_info = ALS_MEDICATIONS.get(med.drug_name, {})
        if drug_info and "pubmed" in drug_info:
            queries.append({
                "topic": f"{med.drug_name} 疗效证据",
                "query": drug_info["pubmed"],
                "priority": 4,
            })

    # 按优先级排序
    queries.sort(key=lambda x: (x["priority"], x["topic"]))
    return queries


# ============================================================================
# 中文文本 → HPO 关键词提取（PhenotypeAgent 用）
# ============================================================================

CHINESE_SYMPTOM_PATTERNS = {
    # (正则模式, HPO术语英, HPO编码)
    "球部": ("球部|延髓|构音|吞咽|饮水呛咳|言语不清|口齿",
             "Bulbar dysfunction", "HP:0002500"),
    "肢体无力": ("无力|乏力|力弱|活动受限|抬举困难",
                "Muscle weakness", "HP:0001324"),
    "肌肉萎缩": ("萎缩|肌肉萎缩|消瘦|变细|凹陷",
                "Muscular atrophy", "HP:0003202"),
    "肉跳": ("肉跳|跳动|颤动|肌束",
             "Fasciculations", "HP:0002380"),
    "呼吸困难": ("呼吸困难|气促|喘|憋气|呼吸衰竭",
                "Respiratory insufficiency", "HP:0002093"),
    "行走困难": ("行走|步行|走路|迈步|下蹲|起立",
                "Gait disturbance", "HP:0001288"),
    "上肢": ("上肢|手臂|手部|手指|腕|肘|肩",
             "Upper limb involvement", "HP:0007354"),
    "下肢": ("下肢|腿部|足|膝|髋|踝",
             "Lower limb involvement", "HP:0007354"),
}
