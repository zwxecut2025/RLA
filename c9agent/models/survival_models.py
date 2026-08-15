"""
c9agent/models/survival_models.py — ALS 生存预测模型

所有模型系数从 data/model_coefficients.json 加载，
可以手动编辑该文件来调整参数，也可以用训练脚本自动拟合。

架构:
  model_coefficients.json  ←─ 手动编辑
       ↑
  CoxPHSurvival.load()     ←─ 推理时自动加载
       ↑
  train_models.py          ←─ 训练脚本写入新系数
"""

import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================================
# 默认系数 —— 当 JSON 文件不存在时的 fallback
# ============================================================================

DEFAULT_COEFFICIENTS = {
    "model_name": "coxph_als",
    "version": "fallback",
    "description": "Literature-derived fallback values",
    "features": [
        "age_per_10yr", "bulbar_onset", "alsfrsr_slope_per_point",
        "fvc_per_10pct", "diagnostic_delay_per_month",
        "c9orf72_positive", "sod1_positive", "male",
    ],
    "coefficients": {
        "age_per_10yr": 0.35,
        "bulbar_onset": 0.42,
        "alsfrsr_slope_per_point": 0.67,
        "fvc_per_10pct": -0.095,
        "diagnostic_delay_per_month": -0.015,
        "c9orf72_positive": 0.47,
        "sod1_positive": 0.22,
        "male": 0.15,
    },
    "baseline_survival": {
        "6": 0.92, "12": 0.80, "18": 0.68, "24": 0.55,
        "30": 0.45, "36": 0.35, "48": 0.22, "60": 0.12,
    },
    "performance": {"c_index": None},
}


# ============================================================================
# 预测结果数据类
# ============================================================================

@dataclass
class SurvivalPrediction:
    """生存预测结果"""
    median_survival_months: float
    survival_prob: dict[int, float]         # {12: 0.85, 24: 0.65, ...}
    risk_level: str                         # "low"/"moderate"/"high"
    c_index: Optional[float] = None
    confidence_interval: Optional[dict] = None


# ============================================================================
# 抽象基类
# ============================================================================

class BaseSurvivalModel(ABC):
    """生存预测模型抽象基类"""

    @abstractmethod
    def predict(self, features: np.ndarray) -> SurvivalPrediction:
        """输入特征向量，返回生存预测"""
        ...

    @abstractmethod
    def get_required_features(self) -> list[str]:
        """返回模型需要的特征列名列表"""
        ...


# ============================================================================
# CoxPH 模型 —— 系数可加载/可编辑/可重训练
# ============================================================================

class CoxPHSurvival(BaseSurvivalModel):
    """
    Cox 比例风险模型。

    核心设计：系数不硬编码，而是从 JSON 文件加载。
    你可以:
    1. 直接编辑 data/model_coefficients.json 微调参数
    2. 运行 scripts/train_models.py 从数据自动拟合
    3. 用 model.save_coefficients() 保存当前参数

    使用:
        model = CoxPHSurvival()                     # 自动加载 JSON
        model = CoxPHSurvival.load("path/to/coef.json")  # 指定文件
        pred = model.predict_from_patient(patient)
    """

    def __init__(self, coefficients_file: str = None):
        """
        参数:
            coefficients_file: 系数 JSON 路径。None = 使用 config 中的路径
        """
        self._coef_file = coefficients_file
        self._coef_data = None
        self._fitted = False
        self.c_index = None

    # —— 系数加载/保存 ——

    @classmethod
    def load(cls, filepath: str = None) -> "CoxPHSurvival":
        """从 JSON 文件加载系数创建模型"""
        model = cls(coefficients_file=filepath)
        model.reload()
        return model

    def reload(self) -> dict:
        """
        重新从 JSON 文件加载系数（热更新）。

        修改 model_coefficients.json 后调用此方法即可生效，无需重启。
        """
        filepath = self._resolve_path()
        if filepath and Path(filepath).exists():
            with open(filepath, "r", encoding="utf-8") as f:
                self._coef_data = json.load(f)
            self._fitted = True
        else:
            print(f"[CoxPH] 系数文件不存在 ({filepath})，使用文献默认值")
            self._coef_data = DEFAULT_COEFFICIENTS
            self._fitted = False
        return self._coef_data

    def save_coefficients(self, filepath: str = None) -> str:
        """保存当前系数到 JSON 文件，返回保存路径"""
        target = filepath or self._resolve_path() or "data/model_coefficients.json"
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._coef_data, f, ensure_ascii=False, indent=2)
        return target

    def _resolve_path(self) -> Optional[str]:
        """解析系数文件路径"""
        if self._coef_file:
            return self._coef_file
        try:
            from c9agent.config import MODEL_CONFIG
            return MODEL_CONFIG.get("coefficients_file")
        except Exception:
            return None

    # —— 属性（从加载的系数中读取） ——

    @property
    def coefficients(self) -> dict:
        """当前系数 {feature_name: log_HR}"""
        if self._coef_data is None:
            self.reload()
        return self._coef_data.get("coefficients", {})

    @property
    def baseline_survival(self) -> dict:
        """基线生存概率 S₀(t)"""
        if self._coef_data is None:
            self.reload()
        return self._coef_data.get("baseline_survival", {})

    @property
    def feature_names(self) -> list[str]:
        if self._coef_data is None:
            self.reload()
        return self._coef_data.get("features", [])

    @property
    def version(self) -> str:
        if self._coef_data is None:
            self.reload()
        return self._coef_data.get("version", "unknown")

    @property
    def metadata(self) -> dict:
        """模型元数据（训练来源、版本等）"""
        if self._coef_data is None:
            self.reload()
        return {
            "model_name": self._coef_data.get("model_name"),
            "version": self._coef_data.get("version"),
            "trained_on": self._coef_data.get("trained_on"),
            "trained_at": self._coef_data.get("trained_at"),
            "description": self._coef_data.get("description"),
            "c_index": self._coef_data.get("performance", {}).get("c_index"),
        }

    # —— 预测 ——

    def predict(self, features: np.ndarray) -> SurvivalPrediction:
        """从特征向量预测"""
        if self._coef_data is None:
            self.reload()
        return self._predict_from_risk_score(features)

    def predict_from_patient(self, patient) -> SurvivalPrediction:
        """从 ALSPatientData 直接预测"""
        if self._coef_data is None:
            self.reload()
        features = self._extract_features(patient)
        return self._predict_from_risk_score(features)

    def _extract_features(self, patient) -> np.ndarray:
        """从 ALSPatientData 提取特征向量（顺序必须与 features 列表一致）"""
        feats = []
        feats.append(max(0, patient.age_at_onset - 60) / 10.0)
        feats.append(1.0 if patient.is_bulbar_onset else 0.0)
        slope = patient.alsfrsr_slope or 0.5
        feats.append(slope)
        fvc = (patient.respiratory.fvc_percent_predicted
               if patient.respiratory else None) or 100
        feats.append((fvc - 100) / 10.0)
        feats.append(patient.diagnostic_delay_months)
        has_c9 = any(
            v.gene == "C9orf72" and v.acmg_classification
            and v.acmg_classification.value in ("P", "LP")
            for v in patient.genetic_variants
        )
        has_sod1 = any(
            v.gene == "SOD1" and v.acmg_classification
            and v.acmg_classification.value in ("P", "LP")
            for v in patient.genetic_variants
        )
        feats.append(1.0 if has_c9 else 0.0)
        feats.append(1.0 if has_sod1 else 0.0)
        feats.append(1.0 if patient.sex.value == "male" else 0.0)
        return np.array(feats)

    def _predict_from_risk_score(self, features: np.ndarray) -> SurvivalPrediction:
        """计算 η = Σ βᵢXᵢ → HR = exp(η) → S(t) = S₀(t)^HR"""
        coef_names = self.feature_names
        coef_values = self.coefficients
        baseline = self.baseline_survival

        # 线性预测器
        log_hr = 0.0
        for i, name in enumerate(coef_names):
            if i < len(features):
                log_hr += coef_values.get(name, 0.0) * features[i]

        hr = np.exp(log_hr)

        # 调整生存概率 —— 保证单调递减
        survival_prob = {}
        prev = 1.0
        for months_str in sorted(baseline.keys(), key=lambda k: int(k)):
            months = int(months_str)
            base_surv = max(0.001, baseline[months_str])  # 避免 0^HR 的不确定性
            adjusted = base_surv ** hr
            # 强制单调递减
            adjusted = min(adjusted, prev)
            survival_prob[months] = round(adjusted, 3)
            prev = adjusted

        median = self._find_median(survival_prob)

        if median < 18:
            risk = "high"
        elif median < 30:
            risk = "moderate"
        else:
            risk = "low"

        ci = median * 0.35

        return SurvivalPrediction(
            median_survival_months=round(median, 1),
            survival_prob=survival_prob,
            risk_level=risk,
            c_index=self._coef_data.get("performance", {}).get("c_index") if self._coef_data else None,
            confidence_interval={
                "lower": round(max(3, median - ci), 1),
                "upper": round(median + ci, 1),
            },
        )

    def _find_median(self, survival_prob: dict[int, float]) -> float:
        """从中位生存概率插值求中位生存期"""
        items = sorted(survival_prob.items())
        for i, (t, s) in enumerate(items):
            if s <= 0.5:
                if i == 0:
                    return float(t)
                t_prev, s_prev = items[i - 1]
                return t_prev + (t - t_prev) * (0.5 - s_prev) / (s - s_prev)
        return float(items[-1][0])

    def get_required_features(self) -> list[str]:
        return self.feature_names

    def __repr__(self):
        return f"<CoxPHSurvival v{self.version} c_index={self.metadata.get('c_index')}>"


# ============================================================================
# CoxPH 训练器 —— 从数据拟合系数
# ============================================================================

class CoxPHTrainer:
    """
    从生存数据拟合 CoxPH 系数。

    使用 lifelines 库的 CoxPHFitter。

    使用:
        trainer = CoxPHTrainer()
        trainer.fit(df, duration_col="survival_months", event_col="event_died")
        trainer.save("data/model_coefficients.json")
    """

    def __init__(self):
        self._fitter = None
        self._summary = None

    def fit(self, df, duration_col: str = "survival_months",
            event_col: str = "event_died",
            feature_cols: list[str] = None) -> "CoxPHTrainer":
        """
        用 lifelines 拟合 CoxPH 模型。

        参数:
            df: pandas DataFrame，含生存时间、事件指示和特征列
            duration_col: 生存时间列名（月）
            event_col: 事件列名（1=死亡, 0=删失）
            feature_cols: 特征列名（None=自动检测所有非duration/event列）
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            raise ImportError(
                "需要 lifelines 库。安装: pip install lifelines"
            )

        if feature_cols is None:
            # 自动排除非特征列
            exclude = {duration_col, event_col, "patient_id", "subject_id"}
            feature_cols = [c for c in df.columns if c not in exclude]

        # 准备数据
        fit_df = df[[duration_col, event_col] + feature_cols].dropna()

        # 拟合
        self._fitter = CoxPHFitter()
        self._fitter.fit(
            fit_df,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=True,
        )

        self._summary = self._fitter.summary
        return self

    def get_coefficients_dict(self) -> dict:
        """提取系数为字典格式"""
        if self._summary is None:
            raise RuntimeError("请先调用 fit() 训练模型")

        coefs = {}
        feature_order = list(self._summary.index)
        for feature in feature_order:
            coefs[feature] = round(float(self._summary.loc[feature, "coef"]), 4)

        return {"coefficients": coefs, "feature_order": feature_order}

    def get_baseline_survival(self) -> dict:
        """提取基线生存函数"""
        if self._fitter is None:
            raise RuntimeError("请先调用 fit() 训练模型")

        baseline = self._fitter.baseline_survival_
        # 转为 {月数: 生存概率} 格式
        result = {}
        for t in baseline.index:
            months = int(round(t))
            if months not in result:  # 取第一个
                result[months] = round(float(baseline.loc[t].iloc[0]), 3)
        return result

    def get_c_index(self) -> float:
        """获取 concordance index"""
        if self._fitter is None:
            raise RuntimeError("请先调用 fit() 训练模型")
        return round(self._fitter.concordance_index_, 3)

    def to_json(self, model_name: str = "coxph_als",
                version: str = "1.0.0",
                trained_on: str = "custom") -> dict:
        """导出为标准系数 JSON 格式"""
        from datetime import date
        coef_data = self.get_coefficients_dict()
        baseline = self.get_baseline_survival()
        c_index = self.get_c_index()

        return {
            "model_name": model_name,
            "version": version,
            "trained_on": trained_on,
            "trained_at": date.today().isoformat(),
            "features": coef_data["feature_order"],
            "coefficients": coef_data["coefficients"],
            "baseline_survival": baseline,
            "performance": {"c_index": c_index},
        }

    def save(self, filepath: str, **meta) -> str:
        """训练完成后保存系数到 JSON"""
        data = self.to_json(**meta)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[CoxPHTrainer] 系数已保存至: {filepath}")
        print(f"  C-index: {data['performance']['c_index']}")
        print(f"  特征数: {len(data['features'])}")
        return filepath

    @property
    def summary(self):
        """lifelines 的模型摘要 DataFrame"""
        return self._summary
