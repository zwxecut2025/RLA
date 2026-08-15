#!/usr/bin/env python
"""
re_extract_als.py — 混合提取：规则提取结构化字段 + LLM 补全语义字段
解决 context 窗口不足的问题
"""
import sys, json, re
from pathlib import Path
from datetime import datetime as dt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c9agent.utils.llm_client import run_llm
from scripts.extract_from_docx import (
    extract_text_from_xlsx, build_patient_json, _fix_date
)


def build_keyvalue_map(text: str) -> dict:
    """
    将 "类别 | 项目 | 结果 | 备注" 格式的文本解析为 key-value 映射。
    同时保留原始行列表。
    """
    kv = {}
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        parts = [p.strip() for p in line.split("|")]
        lines.append(parts)
        if len(parts) >= 3:
            key = parts[1] if len(parts) >= 2 else parts[0]
            value = parts[2] if len(parts) >= 3 else ""
            note = parts[3] if len(parts) >= 4 else ""
            kv[key] = {"value": value, "note": note, "category": parts[0] if parts else ""}
    return kv, lines


def extract_patient_data(text: str) -> dict:
    """混合提取：规则搞定结构化字段，LLM 只做语义理解和摘要"""
    kv, lines = build_keyvalue_map(text)

    result = {
        "patient_id": "",
        "sex": "male",
        "age_at_onset": None,
        "onset_site": "unknown",
        "diagnosis_date": None,
        "symptom_onset_date": None,
        "alsfrsr_records": [],
        "fvc_percent_predicted": None,
        "genetic_variants": [],
        "medications": [],
        "milestones": [],
        "family_history_als": False,
        "clinical_notes": "",
    }

    # —— 1. 基本信息 ——
    for key_part, field in [("姓名", "patient_id"), ("性别", "sex_raw"), ("年龄", "age_raw")]:
        if key_part in kv:
            result[field] = kv[key_part]["value"]
    if "性别" in kv:
        result["sex"] = "female" if "女" in kv["性别"]["value"] else "male"

    # 年龄处理
    if "年龄" in kv:
        age_str = kv["年龄"]["value"]
        m = re.search(r'(\d+)', age_str)
        if m:
            result["age_at_onset"] = int(m.group(1))

    # —— 2. 日期提取 ——
    date_patterns = [
        ("确诊时间", "diagnosis_date"),
        ("首发症状时间", "symptom_onset_date"),
        ("入院日期", "admission_date"),
    ]
    for key_str, field in date_patterns:
        for k, v in kv.items():
            if key_str in k:
                m = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?', v["value"])
                if m:
                    result[field] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                else:
                    m = re.search(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', v["value"])
                    if m:
                        result[field] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 补充: 从其他行中搜索日期
    for k, v in kv.items():
        if "确诊" in k and not result["diagnosis_date"]:
            m = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', v["value"])
            if m:
                result["diagnosis_date"] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-01"

    # —— 3. ALSFRS-R 总分 ——
    # 找 "ALSFRS-R总分" 行
    alsfrsr_total = None
    alsfrsr_date = None
    for k, v in kv.items():
        if "ALSFRS-R总分" in k or ("ALSFRS" in k and "总分" in k):
            m = re.search(r'(\d{1,2})\s*/\s*48', v["value"])
            if m:
                alsfrsr_total = int(m.group(1))

    # 找最近的评估日期
    for k, v in kv.items():
        if "入院日期" in k:
            m = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?', v["value"])
            if m:
                alsfrsr_date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    if alsfrsr_total is not None:
        result["alsfrsr_records"].append({
            "date": alsfrsr_date or dt.today().strftime("%Y-%m-%d"),
            "total_score": alsfrsr_total,
        })

    # 同时收集 ALSFRS-R 子项分数（用于 clinical_notes）
    alsfrsr_subscores = {}
    for k, v in kv.items():
        if k.startswith("ALSFRS-R-"):
            sub_name = k.replace("ALSFRS-R-", "")
            m = re.search(r'(\d+)', v["value"])
            if m:
                alsfrsr_subscores[sub_name] = int(m.group(1))

    # —— 4. FVC ——
    fvc_records = []
    for k, v in kv.items():
        if "FVC" in k.upper():
            m = re.search(r'(\d+(?:\.\d+)?)\s*%', v["value"])
            if m:
                pct = int(float(m.group(1)))
                # 提取日期
                date_str = None
                m2 = re.search(r'(\d{4})[.](\d{1,2})', k)
                if m2:
                    date_str = f"{int(m2.group(1)):04d}-{int(m2.group(2)):02d}-01"
                fvc_records.append({"date": date_str, "fvc_pct": pct})
            else:
                # 数值可能在后一列
                m = re.search(r'(\d+(?:\.\d+)?)\s*L?\s*\(?(\d+)%?\)?', v["value"])
                if m:
                    fvc_records.append({"date": None, "fvc_pct": int(m.group(2))})

    # 取最新的 FVC
    if fvc_records:
        # 按日期排序取最新
        fvc_records.sort(key=lambda x: x["date"] or "", reverse=True)
        result["fvc_percent_predicted"] = fvc_records[0]["fvc_pct"]

    # —— 5. 基因检测 ——
    gene_names = ["SOD1", "C9orf72", "TARDBP", "FUS", "VCP", "PFN1",
                  "TBK1", "OPTN", "UBQLN2", "SQSTM1", "CHCHD10",
                  "NEK1", "KIF5A", "ANXA11", "HNRNPA1", "MATR3", "TUBA4A"]
    for gene in gene_names:
        if gene in kv:
            val = kv[gene]["value"]
            note = kv[gene].get("note", "")
            combined = val + " " + note
            is_wildtype = any(w in combined for w in ["野生型", "野生", "wild", "阴性", "排除", "未检出"])
            if is_wildtype:
                result["genetic_variants"].append({
                    "gene": gene,
                    "variant_type": "wild_type",
                    "acmg_classification": "B",
                    "zygosity": "homozygous_reference",
                })
            else:
                is_pathogenic = any(w in combined for w in ["致病", "pathogenic", "阳性", "突变"])
                result["genetic_variants"].append({
                    "gene": gene,
                    "variant_type": "unknown",
                    "acmg_classification": "P" if is_pathogenic else "VUS",
                    "zygosity": "heterozygous",
                })

    # —— 6. 用药 ——
    med_map = {
        "利鲁唑": "Riluzole", "力如太": "Riluzole", "riluzole": "Riluzole",
        "依达拉奉": "Edaravone", "edaravone": "Edaravone",
        "拉迪卡苷": "Radicava", "radicava": "Radicava",
    }
    for k, v in kv.items():
        for zh, en in med_map.items():
            if zh in k or zh in v["value"]:
                if not any(m["drug_name"] == en for m in result["medications"]):
                    result["medications"].append({
                        "drug_name": en,
                        "dosage": None,
                        "start_date": None,
                    })

    # —— 7. 起病部位推断 ——
    # 从首发症状、临床表现推断
    text_lower = text.lower()
    has_bulbar = any(w in text for w in ["球部", "延髓", "言语", "吞咽", "构音", "含糊", "饮水"])
    has_cervical = any(w in text for w in ["上肢", "手部", "手臂", "颈段"])
    has_lumbar = any(w in text for w in ["下肢", "腿部", "腰段", "足部", "行走"])
    has_respiratory = any(w in text for w in ["呼吸", "呼吸困难"])

    if has_bulbar and has_cervical:
        result["onset_site"] = "bulbar"  # 球部起病更典型
    elif has_bulbar:
        result["onset_site"] = "bulbar"
    elif has_cervical and has_lumbar:
        result["onset_site"] = "spinal_cervical"
    elif has_lumbar:
        result["onset_site"] = "spinal_lumbar"
    elif has_cervical:
        result["onset_site"] = "spinal_cervical"
    elif has_respiratory:
        result["onset_site"] = "respiratory"

    # 从"首发症状"行获取更精确的信息
    for k, v in kv.items():
        if "首发症状" in k:
            onset_text = v["value"] + " " + v.get("note", "")
            if any(w in onset_text for w in ["言语", "吞咽", "球部", "口齿"]):
                result["onset_site"] = "bulbar"
            elif any(w in onset_text for w in ["上肢", "手"]):
                result["onset_site"] = "spinal_cervical"
            elif any(w in onset_text for w in ["下肢", "腿", "行走", "足"]):
                result["onset_site"] = "spinal_lumbar"

    # —— 8. 家族史 ——
    for k, v in kv.items():
        if "家族史" in k or "ALS家族史" in k:
            if any(w in v["value"] for w in ["无", "否认", "阴性", "没有"]):
                result["family_history_als"] = False
            elif any(w in v["value"] for w in ["有", "阳性"]):
                result["family_history_als"] = True

    # —— 9. 临床分期 ——
    # 从诊断标准行提取
    for k, v in kv.items():
        if "确诊级别" in k or "诊断级别" in k:
            result["_diagnosis_level"] = v["value"]

    # —— 10. 里程碑 ——
    for k, v in kv.items():
        if "确诊时间" in k:
            date_val = _fix_date(result.get("diagnosis_date"))
            if date_val and not any(m["milestone_type"] == "diagnosis" for m in result["milestones"]):
                result["milestones"].append({"milestone_type": "diagnosis", "date": date_val})

    # NIV 使用
    for k, v in kv.items():
        if "NIV" in k.upper() or "无创通气" in k:
            result["_niv"] = v["value"]

    # PEG
    for k, v in kv.items():
        if "PEG" in k.upper() or "胃造瘘" in k or "经皮内镜" in k:
            result["_peg"] = v["value"]

    # —— 保存子分数供 LLM 摘要使用 ——
    result["_alsfrsr_subscores"] = alsfrsr_subscores
    result["_fvc_history"] = fvc_records

    return result


def make_llm_summary(extracted: dict) -> str:
    """用 LLM 生成 200 字临床摘要（轻量级，context友好）"""
    summary_data = {
        "姓名": extracted.get("patient_id", ""),
        "性别": extracted.get("sex", ""),
        "发病年龄": extracted.get("age_at_onset", ""),
        "起病部位": extracted.get("onset_site", ""),
        "确诊日期": extracted.get("diagnosis_date", ""),
        "首发症状日期": extracted.get("symptom_onset_date", ""),
        "ALSFRS-R总分": extracted["alsfrsr_records"][0]["total_score"] if extracted["alsfrsr_records"] else "N/A",
        "ALSFRS-R子分数": extracted.get("_alsfrsr_subscores", {}),
        "FVC%": extracted.get("fvc_percent_predicted", ""),
        "FVC历史": extracted.get("_fvc_history", []),
        "基因检测": [f"{v['gene']}:{v['acmg_classification']}" for v in extracted.get("genetic_variants", [])],
        "用药": [m["drug_name"] for m in extracted.get("medications", [])],
        "诊断级别": extracted.get("_diagnosis_level", ""),
    }

    prompt = f"""你是一个 ALS 临床专家。请根据以下患者数据，用200字中文撰写临床摘要。

患者数据:
{json.dumps(summary_data, ensure_ascii=False, indent=2)}

请用一段流畅的中文总结该ALS患者的临床特征、关键指标、基因结果和治疗方案。只返回摘要文本，不要JSON。"""

    try:
        result = run_llm(prompt, max_tokens=500)
        return result.strip()
    except Exception:
        return ""


def main():
    xlsx_path = Path("data/ALS_临床报告.xlsx")
    if not xlsx_path.exists():
        print(f"文件不存在: {xlsx_path}")
        sys.exit(1)

    print(f"提取 Excel: {xlsx_path}")
    text = extract_text_from_xlsx(str(xlsx_path))
    print(f"  文本长度: {len(text)} 字符")

    # Step 1: 规则提取
    print("\n规则提取结构化字段...")
    extracted = extract_patient_data(text)

    # Step 2: LLM 生成摘要
    print("LLM 生成临床摘要...")
    notes = make_llm_summary(extracted)
    extracted["clinical_notes"] = notes

    # Step 3: 构建最终 JSON
    patient_json = build_patient_json(extracted)

    # 清理内部字段
    for k in list(patient_json.keys()):
        if k.startswith("_"):
            del patient_json[k]

    # —— 打印结果 ——
    print("\n" + "=" * 60)
    print("提取结果")
    print("=" * 60)
    fields = [
        ("患者ID", patient_json["patient_id"]),
        ("性别", patient_json["sex"]),
        ("发病年龄", f"{patient_json['age_at_onset']}岁"),
        ("起病部位", patient_json["onset_site"]),
        ("确诊日期", patient_json["diagnosis_date"]),
        ("症状出现", patient_json.get("symptom_onset_date") or "未提及"),
    ]
    for label, value in fields:
        print(f"  {label:12s}: {value}")

    print(f"\n  ALSFRS-R:")
    for r in patient_json["alsfrsr_records"]:
        print(f"    {r['date']}: {r['total_score']}/48")

    print(f"\n  FVC%: {patient_json['respiratory']['fvc_percent_predicted']}%")
    print(f"  基因 ({len(patient_json['genetic_variants'])}个):")
    for v in patient_json["genetic_variants"]:
        print(f"    {v['gene']}: {v['acmg_classification']} ({v['variant_type']})")
    print(f"  用药: {', '.join(m['drug_name'] for m in patient_json['medications'])}")
    print(f"  家族ALS史: {'是' if patient_json.get('family_history_als') else '否'}")
    print(f"  临床摘要: {patient_json.get('clinical_notes', '')[:200]}...")

    # —— 保存 ——
    output_path = Path("data/ALS_patient.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(patient_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n已保存: {output_path}")

    print(f"\n下一步运行预测:")
    print(f"  python scripts/run_single_patient.py --input data/ALS_patient.json -o data/ALS_patient_report.json -m data/ALS_patient_report.md")


if __name__ == "__main__":
    main()
