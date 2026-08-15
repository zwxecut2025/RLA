"""
c9agent/data/patient_schema.py — ALS 患者标准化数据模型

这是整个系统的"通用语言"——所有 Agent 都使用这个数据结构交换信息。
无论原始输入是什么格式（JSON / CSV / 中文自由文本 / VCF文件），
最终都会被 PhenotypeAgent 规范化为这个结构。

设计原则：
1. 使用 Pydantic BaseModel 做自动校验（年龄不能为负、ALSFRS-R 0-48等）
2. 关键临床参数用 @property 自动计算（斜率、分期）
3. 可选字段用 Optional，允许数据不完整
"""

from datetime import date
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, computed_field


# ============================================================================
# 枚举类型 — 限定合法值，防止输入错误
# ============================================================================

class OnsetSite(str, Enum):
    """ALS 发病部位。球部起病预后最差，脊髓起病按节段分"""
    BULBAR = "bulbar"                   # 球部起病（说话/吞咽先受累）
    SPINAL_CERVICAL = "spinal_cervical"  # 颈段起病（上肢先受累）
    SPINAL_LUMBAR = "spinal_lumbar"     # 腰段起病（下肢先受累）
    SPINAL_THORACIC = "spinal_thoracic" # 胸段起病（呼吸先受累，罕见）
    RESPIRATORY = "respiratory"         # 呼吸起病
    UNKNOWN = "unknown"


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class VariantType(str, Enum):
    """基因变异类型"""
    REPEAT_EXPANSION = "repeat_expansion"  # 重复扩增（如C9orf72 GGGGCC）
    MISSENSE = "missense"                  # 错义（氨基酸改变）
    NONSENSE = "nonsense"                  # 无义（提前终止）
    FRAMESHIFT = "frameshift"              # 移码
    SPLICE = "splice"                      # 剪接位点
    DELETION = "deletion"                  # 缺失
    DUPLICATION = "duplication"            # 重复
    UNKNOWN = "unknown"


class Zygosity(str, Enum):
    HETEROZYGOUS = "heterozygous"
    HOMOZYGOUS = "homozygous"
    HEMIZYGOUS = "hemizygous"            # X染色体


class ACMGClass(str, Enum):
    """ACMG 致病性分类"""
    PATHOGENIC = "P"                     # 致病
    LIKELY_PATHOGENIC = "LP"            # 可能致病
    VUS = "VUS"                          # 意义不明
    LIKELY_BENIGN = "LB"                # 可能良性
    BENIGN = "B"                         # 良性


class MilestoneType(str, Enum):
    """ALS 关键临床里程碑事件"""
    DIAGNOSIS = "diagnosis"                   # 确诊
    GASTROSTOMY = "gastrostomy"               # 胃造瘘
    WHEELCHAIR = "wheelchair"                 # 轮椅依赖
    LOSS_OF_AMBULATION = "loss_of_ambulation" # 丧失行走能力
    LOSS_OF_SPEECH = "loss_of_speech"         # 丧失语言能力
    LOSS_OF_SWALLOWING = "loss_of_swallowing" # 丧失吞咽能力
    NIV_DEPENDENCY = "niv_dependency"         # 无创通气依赖 (>22h/天)
    INVASIVE_VENTILATION = "invasive_ventilation"  # 有创通气/气管切开
    DEATH = "death"


# ============================================================================
# 核心数据模型
# ============================================================================

class GeneVariant(BaseModel):
    """单个 ALS 相关基因变异"""
    gene: str                                    # 基因名 e.g. "C9orf72", "SOD1"
    variant_type: VariantType
    hgvs_c: Optional[str] = None                 # cDNA 描述 e.g. "NM_000454.5:c.4A>G"
    hgvs_p: Optional[str] = None                 # 蛋白描述 e.g. "NP_000445.1:p.Ala5Val"
    zygosity: Zygosity
    clinvar_id: Optional[str] = None             # ClinVar 变异编号
    acmg_classification: Optional[ACMGClass] = None
    gnomad_af: Optional[float] = None            # gnomAD 等位基因频率
    pathogenicity_score: Optional[float] = None  # CADD/REVEL 预测分数
    description: Optional[str] = None            # 自由文本描述


class ALSFRSR_Observation(BaseModel):
    """
    单次 ALSFRS-R 测量记录。
    12个功能子项，每项 0(完全丧失)-4(正常)，总分 0-48。
    分数越低 = 功能越差。
    """
    date: date
    total_score: int = Field(ge=0, le=48, description="ALSFRS-R 总分")

    # —— 球部功能 (Bulbar) ——
    speech: Optional[int] = Field(default=None, ge=0, le=4)
    salivation: Optional[int] = Field(default=None, ge=0, le=4)
    swallowing: Optional[int] = Field(default=None, ge=0, le=4)

    # —— 精细运动 (Fine Motor) ——
    handwriting: Optional[int] = Field(default=None, ge=0, le=4)
    cutting_food: Optional[int] = Field(default=None, ge=0, le=4)
    dressing_hygiene: Optional[int] = Field(default=None, ge=0, le=4)

    # —— 粗大运动 (Gross Motor) ——
    turning_in_bed: Optional[int] = Field(default=None, ge=0, le=4)
    walking: Optional[int] = Field(default=None, ge=0, le=4)
    climbing_stairs: Optional[int] = Field(default=None, ge=0, le=4)

    # —— 呼吸功能 (Respiratory) ——
    dyspnea: Optional[int] = Field(default=None, ge=0, le=4)
    orthopnea: Optional[int] = Field(default=None, ge=0, le=4)
    respiratory_insufficiency: Optional[int] = Field(default=None, ge=0, le=4)


class RespiratoryData(BaseModel):
    """呼吸功能数据"""
    fvc_percent_predicted: Optional[float] = None      # FVC % 预计值
    fvc_date: Optional[date] = None
    niv_usage_hours_per_day: Optional[float] = None    # 无创通气 小时/天
    niv_start_date: Optional[date] = None
    invasive_ventilation: bool = False                  # 是否已气管切开


class Medication(BaseModel):
    """用药记录"""
    drug_name: str                            # "Riluzole", "Edaravone", "Tofersen"
    start_date: date
    end_date: Optional[date] = None           # None = 仍在用
    dosage: Optional[str] = None              # e.g. "50mg bid"


class Milestone(BaseModel):
    """临床里程碑事件"""
    milestone_type: MilestoneType
    date: date
    note: Optional[str] = None


# ============================================================================
# 主模型：规范化 ALS 患者数据
# ============================================================================

class ALSPatientData(BaseModel):
    """
    ALS 患者规范化数据 —— 整个系统的"标准通行证"。

    所有 Agent 通过这个结构交换信息。不含任何原始输入格式的细节——
    那些已经在 PhenotypeAgent 阶段被标准化了。
    """
    # —— 基本信息 ——
    patient_id: str
    sex: Sex
    age_at_onset: float = Field(gt=0, le=120, description="发病年龄")

    # —— 发病 ——
    onset_site: OnsetSite
    diagnosis_date: date
    symptom_onset_date: Optional[date] = None        # 症状最早出现日期

    # —— 纵向 ALSFRS-R 记录 ——
    alsfrsr_records: list[ALSFRSR_Observation] = []

    # —— 呼吸 ——
    respiratory: Optional[RespiratoryData] = None

    # —— 遗传 ——
    genetic_variants: list[GeneVariant] = []
    family_history_als: bool = False                  # 家族史

    # —— 用药 ——
    medications: list[Medication] = []

    # —— 里程碑 ——
    milestones: list[Milestone] = []

    # —— 自由文本（从临床笔记提取结构化数据后的原始文本备份）——
    clinical_notes: Optional[str] = None

    # ========================================================================
    # 自动计算的临床参数 (@computed_field 会随输入自动更新)
    # ========================================================================

    @computed_field
    @property
    def latest_alsfrsr(self) -> Optional[ALSFRSR_Observation]:
        """最新的 ALSFRS-R 记录"""
        if not self.alsfrsr_records:
            return None
        return max(self.alsfrsr_records, key=lambda r: r.date)

    @computed_field
    @property
    def alsfrsr_slope(self) -> Optional[float]:
        """
        ALSFRS-R 下降速率 = (48 - 最新分数) / 病程月数
        单位：分/月。这是 ALS 最重要的预后指标。
        > 0.89 分/月 = 快速进展型（文献阈值）
        """
        latest = self.latest_alsfrsr
        onset = self.symptom_onset_date or self.diagnosis_date
        if latest is None:
            return None
        months = (latest.date - onset).days / 30.44
        if months <= 0:
            return None
        return (48 - latest.total_score) / months

    @computed_field
    @property
    def diagnostic_delay_months(self) -> float:
        """诊断延迟（症状出现到确诊的月数）"""
        onset = self.symptom_onset_date or self.diagnosis_date
        return (self.diagnosis_date - onset).days / 30.44

    @computed_field
    @property
    def progression_category(self) -> Optional[str]:
        """进展速度分类"""
        if self.alsfrsr_slope is None:
            return None
        if self.alsfrsr_slope > 0.89:
            return "fast"       # 快速进展
        elif self.alsfrsr_slope > 0.45:
            return "moderate"   # 中等进展
        else:
            return "slow"       # 缓慢进展

    @computed_field
    @property
    def kings_stage(self) -> Optional[int]:
        """
        King's 临床分期 (1-4)
        Stage 1 = 一个区域受累
        Stage 2 = 两个区域
        Stage 3 = 三个区域
        Stage 4 = 需要 NIV 或胃造瘘，或四个区域均受累（呼吸/营养衰竭）

        统一委托给 utils.als_calculators.calc_kings_stage，
        避免与该函数的 off-by-one 计算出现不一致。
        """
        from c9agent.utils.als_calculators import calc_kings_stage
        return calc_kings_stage(self)

    @computed_field
    @property
    def is_bulbar_onset(self) -> bool:
        """是否球部起病（预后较差）"""
        return self.onset_site == OnsetSite.BULBAR

    @computed_field
    @property
    def has_pathogenic_variant(self) -> bool:
        """是否携带已知致病/可能致病变异"""
        return any(
            v.acmg_classification in (ACMGClass.PATHOGENIC, ACMGClass.LIKELY_PATHOGENIC)
            for v in self.genetic_variants
        )


# ============================================================================
# 输入模型 — 接受多种格式的原始输入
# ============================================================================

class ALSPatientInput(BaseModel):
    """
    灵活的输入接口 —— 支持三种输入方式：
    1. 结构化 JSON（已经是 ALSPatientData）
    2. 自由文本（中文/英文临床笔记）
    3. CSV 路径（含纵向 ALSFRS-R 记录）

    PhenotypeAgent 负责将任意输入规范化为 ALSPatientData。
    """
    patient_id: str
    structured_data: Optional[ALSPatientData] = None    # 方式1
    free_text: Optional[str] = None                     # 方式2
    csv_path: Optional[str] = None                      # 方式3
    vcf_path: Optional[str] = None                      # 基因 VCF 文件
