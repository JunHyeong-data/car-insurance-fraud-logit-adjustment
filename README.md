# 자동차 보험 사기 탐지 — Logit Adjustment 적용 및 분석
---

## 📌 연구 개요

자동차 보험 사기 데이터는 사기 비율이 **5.99%** 에 불과한 극심한 클래스 불균형 문제를 안고 있습니다.  
본 프로젝트는 데이터 증강이나 손실함수 변경 없이, 학습된 분류기의 출력에 **사후적으로 Logit Adjustment를 적용**하여 클래스 불균형으로 인한 편향을 완화하는 방법을 검증합니다.

Menon et al. (2021)이 제안한 Logit Adjustment를 자동차 보험 사기 데이터에 적용하고, Baseline 및 Class-Weighted 방식과 비교 분석하였습니다.

---

## 📂 데이터셋

| 항목 | 내용 |
|---|---|
| 출처 | `fraud_oracle.csv` (공개 데이터셋) |
| 규모 | 15,419건 / 33개 변수 |
| 종속변수 | `FraudFound_P` (0: 정상, 1: 사기) |
| 클래스 분포 | 정상 94.01% / 사기 5.99% |

---

## 🔬 실험 설계

### 비교 방법론

| 방법 | 설명 |
|---|---|
| **Baseline** | 보정 없음. 임계값 0.5 적용 |
| **Class-Weighted** | 학습 단계에서 소수 클래스 가중치 부여 |
| **Logit Adjustment (τ=1)** | 학습 후 사후 보정. 사전확률 편향 제거 |

### 비교 모델

- **Elastic Net Logistic Regression** — 선형 판별 모델
- **Random Forest** — 배깅 기반 앙상블
- **CatBoost** — 부스팅 기반 앙상블

### 평가 지표

- Recall, F₂-Score, G-Mean, PR-AUC
- 비대칭 오분류 비용 지표 Cost (C_FN=10, C_FP=1)

---

## 📊 핵심 결과

### Holdout(Test 20%) 성능 비교

| Model | Variant | Recall | F₂ | G-Mean | PR-AUC | Cost |
|---|---|---|---|---|---|---|
| Elastic Net LR | Baseline | 0.0000 | 0.0000 | 0.0000 | 0.1827 | 0.5999 |
| Elastic Net LR | Class_Weighted | 0.9351 | 0.4125 | 0.7438 | 0.1623 | 0.4228 |
| Elastic Net LR | LogitAdj(τ=1) | 0.9297 | 0.4097 | 0.7410 | 0.1827 | 0.4270 |
| Random Forest | Baseline | 0.0108 | 0.0135 | 0.1040 | 0.2453 | 0.5934 |
| Random Forest | Class_Weighted | 0.0054 | 0.0067 | 0.0735 | 0.2464 | 0.5966 |
| Random Forest | LogitAdj(τ=1) | 0.9081 | 0.4138 | 0.7461 | 0.2453 | 0.4189 |
| CatBoost | Baseline | 0.0378 | 0.0467 | 0.1944 | 0.2753 | 0.5781 |
| CatBoost | Class_Weighted | 0.6595 | 0.4580 | 0.7433 | 0.2506 | 0.3567 |
| **CatBoost** | **LogitAdj(τ=1)** | **0.8703** | **0.4792** | **0.7978** | **0.2753** | **0.3304** |

### 5-Fold 교차검증 (CatBoost)

| Variant | μ(Cost) | σ(Cost) | μ(G-Mean) |
|---|---|---|---|
| Baseline | 0.5727 | 0.0068 | 0.2098 |
| Class_Weighted | 0.3814 | 0.0149 | 0.7116 |
| **LogitAdj(τ=1)** | **0.3338** | **0.0082** | **0.7952** |

> **CatBoost + Logit Adjustment** 조합이 Holdout Cost 0.3304로 최저값 달성  
> Baseline 대비 **42.9%** 감소 / Class Weighted 대비 **7.4%** 감소  
> 5-Fold σ(Cost) = 0.0082로 Class Weighted의 절반 수준 — 일반화 안정성 우수

---

## 💡 핵심 발견

**1. Logit Adjustment의 모델 무관 안정성**  
LR, RF, CatBoost 세 모델 모두에서 Recall 87~93%로 일관된 성능을 보였습니다. 단조변환 성질 덕분에 모델 내부 구조와 무관하게 일관되게 작동합니다.

**2. Class-Weighted의 모델 의존성**  
LR과 CatBoost에서는 작동했지만 Random Forest에서는 Recall 0.54%로 오히려 Baseline보다 낮았습니다. 클래스 가중치 방식이 모델 구조에 따라 효과가 달라질 수 있음을 시사합니다.

**3. PR-AUC 보존**  
Baseline과 LogitAdj의 PR-AUC가 모든 모델에서 동일합니다. Logit Adjustment가 모델의 판별력 자체를 변화시키지 않고 결정 경계의 위치만 이동시키기 때문입니다.

**4. 도메인 검증**  
모델이 Fault를 1순위 변수로 학습했으며, 실제로 Policy Holder의 사기율이 Third Party의 **8.9배**로 도메인 지식과 일치합니다.

---

## 📚 참고문헌

- Menon, A.K. et al. (2021). *Long-tail learning via logit adjustment.* ICLR.
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* JAIR.
- Krawczyk, B. (2016). *Learning from imbalanced data.* Progress in Artificial Intelligence.
- Geman, S. et al. (1992). *Neural networks and the bias/variance dilemma.* Neural Computation.
