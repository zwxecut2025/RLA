#!/usr/bin/env python
"""
run_api_server.py — C9Agent FastAPI 服务

提供 REST API 接口，供临床系统/HIS 调用。

启动:
    python scripts/run_api_server.py

端点:
    POST /analyze        单患者分析
    POST /cohort         队列分析
    GET  /health         健康检查
    GET  /docs           Swagger 文档
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("需要安装 fastapi: pip install fastapi uvicorn")
    sys.exit(1)

from c9agent.core.orchestrator import CentralOrchestrator
from c9agent.data.patient_schema import ALSPatientInput
from c9agent.data.synthetic_data import SyntheticALSGenerator
from c9agent.agents.cohort_agent import CohortAgent
from c9agent.models.survival_models import CoxPHSurvival


# ============================================================================
# 请求/响应模型
# ============================================================================

class PatientRequest(BaseModel):
    """单患者分析请求"""
    patient_id: str
    sex: Optional[str] = "male"
    age_at_onset: Optional[float] = 55
    onset_site: Optional[str] = "unknown"
    alsfrsr_total: Optional[int] = None
    alsfrsr_slope: Optional[float] = None
    fvc_percent: Optional[float] = None
    gene_variants: Optional[list[str]] = []
    clinical_notes: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P001",
                "sex": "male",
                "age_at_onset": 64,
                "onset_site": "bulbar",
                "alsfrsr_total": 34,
                "alsfrsr_slope": 1.2,
                "fvc_percent": 65,
                "gene_variants": ["C9orf72"],
                "clinical_notes": "球部起病，构音障碍+吞咽困难，快速进展"
            }
        }


class CohortRequest(BaseModel):
    """队列分析请求"""
    patients: list[PatientRequest]
    survival_data: Optional[dict[str, float]] = None
    events: Optional[dict[str, bool]] = None


# ============================================================================
# 应用
# ============================================================================

app = FastAPI(
    title="C9Agent — ALS Intelligent Analysis API",
    description="基于 DeepRare 多层智能体架构的 ALS 预后分析与队列挖掘服务",
    version="0.4.0",
)

# CORS（允许前端跨域调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局（懒加载）
_orchestrator: Optional[CentralOrchestrator] = None
_cohort_agent: Optional[CohortAgent] = None
_survival_model: Optional[CoxPHSurvival] = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CentralOrchestrator(enable_reflection=True)
    return _orchestrator


def get_cohort_agent():
    global _cohort_agent
    if _cohort_agent is None:
        _cohort_agent = CohortAgent()
    return _cohort_agent


# ============================================================================
# 端点
# ============================================================================

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "0.4.0",
        "timestamp": datetime.now().isoformat(),
        "agents": ["PhenotypeAgent", "PrognosisAgent", "LiteratureAgent", "CohortAgent"],
    }


@app.post("/analyze")
async def analyze_single_patient(req: PatientRequest):
    """
    对单个 ALS 患者进行完整分析。

    包含:
    - 表型提取与分期
    - 生存预测 (CoxPH)
    - 文献证据检索
    - 自我反思循环
    - 可追溯推理链
    """
    try:
        # 构造输入
        from c9agent.data.patient_schema import (
            ALSPatientData, OnsetSite, Sex, ALSFRSR_Observation,
            GeneVariant, VariantType, Zygosity, ACMGClass,
        )
        from datetime import date

        onset_map = {
            "bulbar": OnsetSite.BULBAR,
            "spinal_cervical": OnsetSite.SPINAL_CERVICAL,
            "spinal_lumbar": OnsetSite.SPINAL_LUMBAR,
            "respiratory": OnsetSite.RESPIRATORY,
            "unknown": OnsetSite.UNKNOWN,
        }

        patient = ALSPatientData(
            patient_id=req.patient_id,
            sex=Sex.MALE if req.sex == "male" else Sex.FEMALE,
            age_at_onset=req.age_at_onset or 55,
            onset_site=onset_map.get(req.onset_site, OnsetSite.UNKNOWN),
            diagnosis_date=date.today(),
        )

        if req.alsfrsr_total is not None:
            patient.alsfrsr_records = [
                ALSFRSR_Observation(date=date.today(), total_score=req.alsfrsr_total)
            ]

        for gene_name in (req.gene_variants or []):
            patient.genetic_variants.append(
                GeneVariant(
                    gene=gene_name,
                    variant_type=VariantType.UNKNOWN,
                    zygosity=Zygosity.HETEROZYGOUS,
                    acmg_classification=ACMGClass.PATHOGENIC,
                )
            )

        input_data = ALSPatientInput(
            patient_id=req.patient_id,
            structured_data=patient,
            free_text=req.clinical_notes,
        )

        # 运行分析
        orch = get_orchestrator()
        report = orch.analyze_single_patient(input_data, verbose=False)

        return {
            "status": "success",
            "report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cohort")
async def analyze_cohort(req: CohortRequest):
    """
    对 ALS 患者队列进行统计分析。

    包含:
    - 队列描述统计
    - 基因型-表型关联
    - Kaplan-Meier 生存曲线
    - 亚组（起病部位/基因型）比较
    """
    try:
        from c9agent.data.patient_schema import ALSPatientData, OnsetSite, Sex
        from datetime import date

        patients = []
        for r in req.patients:
            onset_map = {
                "bulbar": OnsetSite.BULBAR,
                "spinal_cervical": OnsetSite.SPINAL_CERVICAL,
                "unknown": OnsetSite.UNKNOWN,
            }
            p = ALSPatientData(
                patient_id=r.patient_id,
                sex=Sex.MALE if r.sex == "male" else Sex.FEMALE,
                age_at_onset=r.age_at_onset or 55,
                onset_site=onset_map.get(r.onset_site, OnsetSite.UNKNOWN),
                diagnosis_date=date.today(),
            )
            patients.append(p)

        agent = get_cohort_agent()
        result = agent.execute(
            cohort=patients,
            survival_data=req.survival_data,
            events=req.events,
        )

        return {
            "status": "success",
            "result": {
                "cohort_size": result.data.get("cohort_size"),
                "description": result.data.get("description"),
                "gene_phenotype_associations": result.data.get("gene_phenotype_associations"),
                "stratified_survival": result.data.get("stratified_survival"),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/demo")
async def demo_report():
    """用合成数据生成一份演示报告"""
    gen = SyntheticALSGenerator(seed=42)
    patient = gen.generate_patient(months_since_onset=18)

    from c9agent.data.patient_schema import ALSPatientInput
    input_data = ALSPatientInput(
        patient_id=patient.patient_id,
        structured_data=patient,
    )

    orch = get_orchestrator()
    report = orch.analyze_single_patient(input_data, verbose=False)

    return {
        "status": "success",
        "patient_id": patient.patient_id,
        "report": report,
    }


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════╗
    ║   C9Agent — ALS 智能分析 API 服务       ║
    ║   启动: http://localhost:8000            ║
    ║   文档: http://localhost:8000/docs       ║
    ╚══════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
