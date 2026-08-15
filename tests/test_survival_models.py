"""
测试生存预测模型
"""
import numpy as np
from c9agent.models.survival_models import CoxPHSurvival


class TestCoxPHSurvival:
    """CoxPH 基线模型测试"""

    def test_predict_basic(self):
        model = CoxPHSurvival()
        # 构造一个"中等"患者: 年龄60, 球部起病, 斜率0.7, FVC=80, 诊延12月, 无基因
        features = np.array([0.0, 1.0, 0.7, -2.0, 12.0, 0.0, 0.0, 1.0])
        pred = model.predict(features)
        assert 3 <= pred.median_survival_months <= 60
        assert len(pred.survival_prob) > 0
        assert all(0 <= p <= 1 for p in pred.survival_prob.values())
        assert pred.risk_level in ("low", "moderate", "high")

    def test_predict_from_patient(self, sample_patient, bulbar_patient):
        model = CoxPHSurvival()
        # 脊髓起病
        pred1 = model.predict_from_patient(sample_patient)
        # 球部起病（预后更差）
        pred2 = model.predict_from_patient(bulbar_patient)
        print(f"\n脊髓起病中位生存: {pred1.median_survival_months}月 (风险={pred1.risk_level})")
        print(f"球部起病中位生存: {pred2.median_survival_months}月 (风险={pred2.risk_level})")
        assert isinstance(pred1.median_survival_months, float)
        assert isinstance(pred2.median_survival_months, float)

    def test_high_risk_patient(self):
        """高风险患者: 高龄 + 球部 + 快速进展 + 低FVC + C9orf72"""
        model = CoxPHSurvival()
        features = np.array([2.5, 1.0, 1.5, -5.0, 6.0, 1.0, 0.0, 1.0])
        pred = model.predict(features)
        print(f"\n高风险中位生存: {pred.median_survival_months}月")
        assert pred.risk_level == "high"

    def test_low_risk_patient(self):
        """低风险患者: 年轻 + 脊髓 + 缓慢进展 + 正常FVC + 无致病变异"""
        model = CoxPHSurvival()
        features = np.array([-2.0, 0.0, 0.2, 0.0, 24.0, 0.0, 0.0, 0.0])
        pred = model.predict(features)
        print(f"\n低风险中位生存: {pred.median_survival_months}月")
        assert pred.risk_level in ("low", "moderate")

    def test_survival_probability_decreases(self):
        """验证生存概率随时间递减"""
        model = CoxPHSurvival()
        features = np.array([0.0, 0.0, 0.5, 0.0, 12.0, 0.0, 0.0, 0.0])
        pred = model.predict(features)
        probs = list(pred.survival_prob.values())
        for i in range(1, len(probs)):
            assert probs[i] <= probs[i-1], f"生存概率应在 {i} 处递减"
