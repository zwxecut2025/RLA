"""
c9agent/data/vector_store.py — 患者向量存储和相似检索

对应 DeepRare 的 Case Searcher:
"Case searcher agent is tasked with exploring an external case bank. Each patient
 is represented as a list of HPO terms, transforming case search into an HPO
 similarity matching problem."

Phase 3 实现: 将患者特征向量化 → 存储在向量数据库中 → 按相似度检索。

技术栈: ChromaDB（嵌入式向量数据库）+ sentence-transformers（嵌入模型）
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class SimilarPatient:
    """一个相似患者记录"""
    patient_id: str
    similarity_score: float                         # 0-1，越高越相似
    age_at_onset: float = 0
    onset_site: str = ""
    alsfrsr_slope: float = 0
    survival_months: float = 0                      # 已知的生存期
    progression_category: str = ""
    risk_level: str = ""
    key_similarities: list[str] = field(default_factory=list)  # 相似的特征
    key_differences: list[str] = field(default_factory=list)   # 不同的特征


# ============================================================================
# 特征提取 —— 患者 → 向量
# ============================================================================

class PatientVectorizer:
    """
    将 ALSPatientData 转换为数值特征向量。

    特征维度（Phase 3: 手工特征，Phase 4: 可用 MedCPT 嵌入）:
    0. age_normalized:         (age - 55) / 15
    1. bulbar_onset:           0/1
    2. alsfrsr_slope:          ΔFS 分/月
    3. fvc_normalized:         (fvc - 80) / 30
    4. diagnostic_delay_months:月
    5. has_c9orf72:            0/1
    6. has_sod1:               0/1
    7. has_tardbp:             0/1
    8. has_fus:                0/1
    9. male:                   0/1
    10. kings_stage:           1-4 (归一化)
    11. progression_category:  0=slow, 1=moderate, 2=fast
    """

    FEATURE_NAMES = [
        "age_normalized", "bulbar_onset", "alsfrsr_slope",
        "fvc_normalized", "diagnostic_delay_months",
        "has_c9orf72", "has_sod1", "has_tardbp", "has_fus",
        "male", "kings_stage", "progression_speed",
    ]

    DIM = len(FEATURE_NAMES)

    def vectorize(self, patient) -> np.ndarray:
        """患者 → 特征向量"""
        feats = []

        # 年龄
        feats.append((patient.age_at_onset - 55) / 15)

        # 起病
        feats.append(1.0 if patient.is_bulbar_onset else 0.0)

        # 进展速率
        slope = patient.alsfrsr_slope or 0.5
        feats.append(slope)

        # FVC
        fvc = (patient.respiratory.fvc_percent_predicted
               if patient.respiratory else 90)
        feats.append((fvc - 80) / 30)

        # 诊断延迟
        feats.append(patient.diagnostic_delay_months / 12.0)

        # 基因
        gene_set = {v.gene for v in patient.genetic_variants}
        feats.append(1.0 if "C9orf72" in gene_set else 0.0)
        feats.append(1.0 if "SOD1" in gene_set else 0.0)
        feats.append(1.0 if "TARDBP" in gene_set else 0.0)
        feats.append(1.0 if "FUS" in gene_set else 0.0)

        # 性别
        feats.append(1.0 if patient.sex.value == "male" else 0.0)

        # 分期
        kings = patient.kings_stage or 2
        feats.append((kings - 1) / 3.0)

        # 进展速度
        speed_map = {"slow": 0.0, "moderate": 0.5, "fast": 1.0, None: 0.5}
        feats.append(speed_map.get(patient.progression_category, 0.5))

        return np.array(feats, dtype=np.float32)

    def vector_to_dict(self, vec: np.ndarray) -> dict:
        """向量 → 特征字典（用于解释相似性）"""
        return {name: round(float(vec[i]), 3)
                for i, name in enumerate(self.FEATURE_NAMES)}


# ============================================================================
# 向量存储
# ============================================================================

class PatientVectorStore:
    """
    患者向量库 —— 存储 + 检索。

    Phase 3 使用内存字典（无外部依赖）。
    生产环境可替换为 ChromaDB / Qdrant。
    """

    def __init__(self):
        self._vectorizer = PatientVectorizer()
        self._patients: dict[str, dict] = {}       # patient_id → {vector, metadata}
        self._vectors: list[np.ndarray] = []        # 所有向量的数组
        self._ids: list[str] = []                   # 向量对应的 ID 列表

    @property
    def vectorizer(self) -> PatientVectorizer:
        return self._vectorizer

    def add_patient(self, patient, survival_months: float = None,
                    outcome: str = None) -> str:
        """
        将患者加入向量库。

        参数:
            patient: ALSPatientData
            survival_months: 已知的生存期（如果有）
            outcome: 结局描述
        """
        pid = patient.patient_id
        vec = self._vectorizer.vectorize(patient)

        self._patients[pid] = {
            "vector": vec,
            "metadata": {
                "patient_id": pid,
                "age": patient.age_at_onset,
                "onset_site": patient.onset_site.value,
                "alsfrsr_slope": patient.alsfrsr_slope,
                "progression": patient.progression_category,
                "kings_stage": patient.kings_stage,
                "survival_months": survival_months,
                "outcome": outcome,
            },
        }

        self._vectors.append(vec)
        self._ids.append(pid)
        return pid

    def add_cohort(self, patients: list, survival_data: dict = None) -> int:
        """批量添加患者队列"""
        count = 0
        for p in patients:
            surv = survival_data.get(p.patient_id) if survival_data else None
            self.add_patient(p, survival_months=surv)
            count += 1
        return count

    def search(self, query_patient,
               top_k: int = 5) -> list[SimilarPatient]:
        """
        检索最相似的患者。

        参数:
            query_patient: ALSPatientData（查询患者）
            top_k: 返回前 K 个最相似的患者

        返回:
            SimilarPatient 列表，按相似度降序
        """
        query_vec = self._vectorizer.vectorize(query_patient)

        if not self._vectors:
            return []

        # 余弦相似度
        all_vecs = np.array([v for v in self._vectors])
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        vecs_norm = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(vecs_norm, query_norm)

        # 排除自身
        top_indices = np.argsort(similarities)[::-1]
        results = []

        for idx in top_indices[:top_k * 2]:  # 多取一些做过滤
            pid = self._ids[idx]
            if pid == query_patient.patient_id:
                continue  # 排除自身

            sim = float(similarities[idx])
            meta = self._patients[pid]["metadata"]

            # 比较特征找异同
            patient_vec = self._patients[pid]["vector"]
            diffs = np.abs(query_vec - patient_vec)

            # 找出相似和不同的特征
            sim_features = []
            diff_features = []
            for i, name in enumerate(self._vectorizer.FEATURE_NAMES):
                if diffs[i] < 0.2:   # 差异 < 0.2 视为相似
                    sim_features.append(name)
                elif diffs[i] > 0.6: # 差异 > 0.6 视为不同
                    diff_features.append(name)

            results.append(SimilarPatient(
                patient_id=pid,
                similarity_score=round(sim, 4),
                age_at_onset=meta.get("age", 0),
                onset_site=meta.get("onset_site", ""),
                alsfrsr_slope=meta.get("alsfrsr_slope", 0),
                survival_months=meta.get("survival_months", 0) or 0,
                progression_category=meta.get("progression", ""),
                risk_level=meta.get("risk_level", ""),
                key_similarities=sim_features[:5],
                key_differences=diff_features[:5],
            ))

            if len(results) >= top_k:
                break

        return results

    def get_cohort_stats(self) -> dict:
        """获取库中患者队列的统计信息"""
        if not self._patients:
            return {"total": 0}

        ages = [m["metadata"]["age"] for m in self._patients.values()]
        slopes = [m["metadata"]["alsfrsr_slope"] or 0 for m in self._patients.values()]
        survivals = [m["metadata"]["survival_months"] or 0 for m in self._patients.values()
                     if m["metadata"]["survival_months"]]

        bulbar_count = sum(
            1 for m in self._patients.values()
            if m["metadata"]["onset_site"] == "bulbar"
        )

        return {
            "total": len(self._patients),
            "mean_age": round(np.mean(ages), 1) if ages else 0,
            "mean_slope": round(np.mean(slopes), 2) if slopes else 0,
            "mean_survival": round(np.mean(survivals), 1) if survivals else 0,
            "bulbar_pct": round(bulbar_count / len(self._patients) * 100, 1),
        }


# ============================================================================
# 全局向量库（单例，跨 Agent 共享）
# ============================================================================

_global_store: Optional[PatientVectorStore] = None


def get_vector_store() -> PatientVectorStore:
    """获取全局向量库（懒加载）"""
    global _global_store
    if _global_store is None:
        _global_store = PatientVectorStore()
    return _global_store
