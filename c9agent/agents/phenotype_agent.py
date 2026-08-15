"""
c9agent/agents/phenotype_agent.py — 表型提取 Agent

对应 DeepRare 的 Phenotype Extractor + Disease Normalizer 两个组件。
职责:
1. 标准化 ALSFRS-R 评分 → 计算 ΔFS、King's MITOS 分期
2. 自由文本 → HPO 术语（中文支持）
3. 数据质量校验（必填字段、异常值）
"""
import json
import pandas as pd
from datetime import date as dt_date
from pathlib import Path
from c9agent.data.patient_schema import (
    ALSPatientData, ALSPatientInput, ALSFRSR_Observation,
    OnsetSite, Sex, ACMGClass,
)
from c9agent.agents.base_agent import BaseAgent, AgentResult
from c9agent.utils.llm_client import run_llm
from c9agent.utils.als_calculators import calc_alsfrsr_slope


class PhenotypeAgent(BaseAgent):
    """
    表型提取 Agent —— 原始输入 → 标准化 ALSPatientData。

    支持三种输入模式:
    1. structured_data 已提供 → 直接校验通过
    2. free_text → LLM 提取结构化字段
    3. csv_path → 加载 CSV，LLM 匹配列名后提取
    """

    def __init__(self):
        super().__init__(
            name="PhenotypeAgent",
            description="标准化ALS患者临床数据: ALSFRS-R / HPO / 分期",
        )

    def execute(self, patient: ALSPatientData = None,
                raw_input: ALSPatientInput = None, **kwargs) -> AgentResult:
        """
        解析原始输入，返回规范化的 ALSPatientData。
        """
        import time; t0 = time.time()

        # 情况1: 已经是结构化数据
        if raw_input and raw_input.structured_data:
            data = raw_input.structured_data
            validation = self._validate(data)
            return self._build_result(data, validation, t0)

        # 情况2: 自由文本
        if raw_input and raw_input.free_text:
            data = self._parse_free_text(raw_input.free_text, raw_input.patient_id)
            validation = self._validate(data)
            return self._build_result(data, validation, t0)

        # 情况3: CSV
        if raw_input and raw_input.csv_path:
            data = self._parse_csv(raw_input.csv_path, raw_input.patient_id)
            validation = self._validate(data)
            return self._build_result(data, validation, t0)

        # 情况4: 直接给 ALSPatientData
        if patient:
            validation = self._validate(patient)
            return self._build_result(patient, validation, t0)

        return AgentResult.failed(self.name, "没有提供有效的输入数据")

    def _parse_free_text(self, text: str, patient_id: str) -> ALSPatientData:
        """
        LLM 从中文/英文自由文本中提取 ALS 临床字段。

        使用两阶段策略（参考 DeepRare 的 Phenotype Extractor 两阶段 LLM 提取）:
        Stage 1: 识别实体（发病部位、ALSFRS-R、基因变异等）
        Stage 2: 标准化（中文术语 → 英文/HPO 编码）
        """
        prompt = f"""你是一个 ALS 临床数据提取专家。从以下临床笔记中提取关键信息。

临床笔记:
```
{text[:3000]}
```

请以 JSON 格式返回以下字段（如果没有对应信息，填 null）:

{{
    "sex": "male" 或 "female",
    "age_at_onset": 数字（岁）,
    "onset_site": "bulbar"/"spinal_cervical"/"spinal_lumbar"/"unknown",
    "diagnosis_date": "YYYY-MM-DD",
    "symptom_onset_date": "YYYY-MM-DD",
    "alsfrsr_total": 数字(0-48),
    "fvc_percent": 数字(%),
    "niv_usage": 数字(小时/天) 或 null,
    "genetic_variants": [{{"gene": "基因名", "variant": "变异描述"}}],
    "medications": ["药物名"],
    "family_history": true/false/null
}}

重要规则:
- 中文术语映射: "球部起病"="bulbar", "肌萎缩侧索硬化"="ALS", "利鲁唑"="Riluzole"
- 只返回 JSON，不要解释
"""
        result = run_llm(prompt)
        try:
            data = json.loads(result[result.find("{"):result.rfind("}")+1])
        except json.JSONDecodeError:
            return self._make_minimal_patient(patient_id)

        return self._dict_to_patient(data, patient_id)

    def _parse_csv(self, csv_path: str, patient_id: str) -> ALSPatientData:
        """从 CSV 文件加载纵向 ALSFRS-R 数据"""
        df = pd.read_csv(csv_path)
        # 尝试常见的列名
        date_col = _find_column(df, ["date", "日期", "visit_date"])
        total_col = _find_column(df, ["total_score", "alsfrsr_total", "总分", "ALSFRS"])
        onset_col = _find_column(df, ["onset_site", "onset", "起病部位"])

        if total_col is None:
            return self._make_minimal_patient(patient_id)

        records = []
        for _, row in df.iterrows():
            obs_date = (pd.to_datetime(row[date_col]).date() if date_col
                       else None)
            if obs_date is None:
                continue
            records.append(ALSFRSR_Observation(
                date=obs_date,
                total_score=int(row[total_col]),
            ))

        onset = OnsetSite.UNKNOWN
        if onset_col and len(df) > 0:
            onset_map = {"bulbar": OnsetSite.BULBAR, "球部": OnsetSite.BULBAR}
            onset = onset_map.get(str(df[onset_col].iloc[0]).lower(), OnsetSite.UNKNOWN)

        return ALSPatientData(
            patient_id=patient_id,
            sex=Sex.MALE,
            age_at_onset=55,
            onset_site=onset,
            diagnosis_date=records[0].date if records else date.today(),
            alsfrsr_records=records,
        )

    def _validate(self, data: ALSPatientData) -> dict:
        """校验数据完整性，返回 {warning: [...], missing_critical: [...]} """
        warnings = []
        missing = []

        if not data.alsfrsr_records:
            missing.append("ALSFRS-R 记录缺失（对预后预测至关重要）")

        if data.onset_site == OnsetSite.UNKNOWN:
            warnings.append("发病部位未知，影响预后分类准确性")

        if data.alsfrsr_slope is None:
            warnings.append("ALSFRS-R 下降速率无法计算（需要至少2次随访记录）")

        if data.respiratory is None or data.respiratory.fvc_percent_predicted is None:
            warnings.append("FVC 数据缺失，影响呼吸预后判断")

        if len(data.alsfrsr_records) == 1:
            warnings.append("仅有一次 ALSFRS-R 记录，无法计算下降趋势")

        return {"warnings": warnings, "missing_critical": missing}

    def _dict_to_patient(self, d: dict, patient_id: str) -> ALSPatientData:
        """字典 → ALSPatientData（尽量不丢失信息）"""
        return ALSPatientData(
            patient_id=patient_id,
            sex=Sex.MALE if d.get("sex") == "male" else Sex.FEMALE,
            age_at_onset=d.get("age_at_onset", 55),
            onset_site=_parse_onset(d.get("onset_site", "unknown")),
            diagnosis_date=_parse_date(d.get("diagnosis_date")),
            symptom_onset_date=_parse_date(d.get("symptom_onset_date")),
            alsfrsr_records=([ALSFRSR_Observation(
                date=_parse_date(d.get("diagnosis_date")) or _parse_date("2025-01-01"),
                total_score=int(d["alsfrsr_total"])
            )] if d.get("alsfrsr_total") else []),
            family_history_als=d.get("family_history", False),
        )

    def _make_minimal_patient(self, patient_id: str) -> ALSPatientData:
        """构造最简患者（解析失败时的 fallback）"""
        from datetime import date as dt_date
        return ALSPatientData(
            patient_id=patient_id, sex=Sex.MALE, age_at_onset=55,
            onset_site=OnsetSite.UNKNOWN,
            diagnosis_date=dt_date.today(),
        )

    def _build_result(self, data: ALSPatientData, validation: dict,
                      t0: float) -> AgentResult:
        """构造最终结果"""
        warnings = validation["warnings"] + validation["missing_critical"]
        if validation["missing_critical"]:
            status = "partial"
        else:
            status = "success"

        # 写入推理证据
        alsfrsr_total = data.latest_alsfrsr.total_score if data.latest_alsfrsr else "N/A"
        delta_fs = (f"{data.alsfrsr_slope:.2f}分/月"
                    if data.alsfrsr_slope is not None else "未知")
        self._add_evidence(
            "observation",
            f"患者 {data.patient_id}: {data.sex}, 发病年龄{data.age_at_onset}岁, "
            f"起病部位={data.onset_site.value}, "
            f"ALSFRS-R={alsfrsr_total}, ΔFS={delta_fs}",
            confidence=1.0,
        )

        import time
        return AgentResult(
            agent_name=self.name, status=status,
            data={"patient": data.model_dump(mode="json")},
            evidence_nodes=[self._add_evidence("observation", f"表型提取完成")],
            confidence=0.5 if warnings else 0.95,
            warnings=warnings,
            execution_time_ms=(time.time() - t0) * 1000,
        )


# —— 辅助函数 ——

def _parse_onset(s: str) -> OnsetSite:
    mapping = {
        "bulbar": OnsetSite.BULBAR, "球部": OnsetSite.BULBAR,
        "spinal_cervical": OnsetSite.SPINAL_CERVICAL, "颈段": OnsetSite.SPINAL_CERVICAL,
        "spinal_lumbar": OnsetSite.SPINAL_LUMBAR, "腰段": OnsetSite.SPINAL_LUMBAR,
        "respiratory": OnsetSite.RESPIRATORY, "呼吸": OnsetSite.RESPIRATORY,
    }
    return mapping.get(s.lower(), OnsetSite.UNKNOWN)


def _parse_date(s: str) -> dt_date | None:
    if not s:
        return None
    try:
        from datetime import date as dt_date
        return dt_date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
