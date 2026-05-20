# 통계적 기계학습 기반 자동차 보험 사기 탐지 분류 모델 비교 연구

### Logit Adjustment 적용 및 분석
*Logit Adjustment for Insurance Fraud Detection under Severe Class Imbalance*

---
## Abstract

본 연구는 자동차 보험 사기 데이터의 극심한 클래스 불균형(사기 비율 5.99%) 환경에서, 인위적 데이터 증강 없이 학습된 분류기의 출력에 사후적으로 **Logit Adjustment**를 적용하여 클래스 불균형으로 인한 편향을 완화하는 방법을 검증한다. Menon et al. (2021)이 제안한 Logit Adjustment를 자동차 보험 사기 데이터에 적용하고, Baseline 및 Class-Weighted 방식과 비교하였다. Stratified 5-Fold 교차검증과 Test 20% Holdout 평가를 결합하여, Elastic Net Logistic Regression, Random Forest, CatBoost 세 모델에 세 가지 보정 방법을 적용한 9개 조합의 성능을 비교하였다. 비대칭 오분류 비용(C_FN=10, C_FP=1)을 반영한 경험적 비용 지표(Cost)를 Recall, F2, G-Mean, PR-AUC와 함께 평가 지표로 사용하였다.

**Keywords** — 보험 사기 탐지, Logit Adjustment, 클래스 불균형, 비대칭 오분류 비용 지표, 편향-분산 트레이드오프

---

## 1. Introduction

자동차 보험 사기는 고의적 사고 조작, 피해 금액 부풀리기, 허위 청구 등 다양한 형태로 발생하며 보험사의 재정 건전성을 위협하고 선량한 가입자들의 보험료 인상을 초래하는 사회적 문제이다. 본 연구에 사용된 데이터는 사기 비율이 약 5.99%에 불과한 극심한 클래스 불균형(Class Imbalance) 문제를 안고 있다. 이 경우 모든 청구를 "정상"으로 분류하는 모델조차 90% 이상의 정확도를 달성할 수 있어, 단순 정확도(Accuracy)는 실제 탐지 성능을 제대로 반영하지 못한다.

본 연구의 주요 목적은 다음과 같다.

1. 데이터에 대한 체계적 탐색 및 전처리 수행
2. 데이터 증강 없이 학습된 모델의 출력만을 사후 보정하는 Logit Adjustment를 적용하여 클래스 불균형에 대응
3. 사기 미탐지 비용이 정상 오탐지 비용보다 크다는 비즈니스 현실을 반영한 비대칭 오분류 비용 지표(Cost)를 평가 지표로 설정하여 비교

---

## 2. Data

분석에 사용한 데이터는 `fraud_oracle.csv`로, 미국 자동차 보험 청구 기록을 바탕으로 구성된 공개 데이터셋이다.

| 항목 | 값 |
|---|---|
| 전체 건수 | 15,419건 (오류 행 1건 제거 후) |
| 변수 수 | 33개 (종속변수 1개 + 독립변수 32개) |
| 정상 청구 | 14,496건 (94.01%) |
| 사기 청구 | 923건 (5.99%) |
| Train / Test 분할 | 80% / 20% Stratified |
| 최종 학습 행렬 | 12,335 × 67 (Train), 3,084 × 67 (Test) |

### 2.1 주요 전처리 절차

1. `DayOfWeekClaimed='0'` 1건 제거
2. `PolicyNumber` 제거 (데이터 누수 위험)
3. `Age=0`인 319건을 16.5로 보정
4. 이진 변환, 순서형 인코딩, 구간 중간값 대체
5. 시간 변수의 sin/cos 순환 변환
6. 명목형 변수의 원-핫 인코딩
7. StandardScaler 표준화

---

## 3. Methodology

### 3.1 Logit Adjustment

학습 분포의 logit은 Bayes 정리에 의해 클래스 조건부 우도비와 사전확률 log-odds의 합으로 분해된다.

> *f*(x) = log[ P(x|y=1) / P(x|y=0) ] + log[ π₁ / π₀ ]

평가 환경에서 사전확률이 균형(π₁ = π₀ = 0.5)이라면 두 번째 항은 0이 되어야 하지만, 본 연구의 학습 분포에서는 π₁ = 0.0598이므로 logit에 log(0.0598/0.9402) ≈ **−2.755**의 음의 편향이 더해진다. 이를 사후에 제거하는 것이 Logit Adjustment의 본질이다.

> *f*_adj(x) = *f*(x) − τ · log( π₁ / π₀ )

τ = 1을 기본값으로 사용한다.

### 3.2 비교 방법론

| 방법 | 보정 단계 | 설명 |
|---|---|---|
| Baseline | 없음 | 표준 분류기, 임계값 0.5 |
| Class Weighted | 학습 단계 | 손실함수에서 소수 클래스 가중치 1/π_y 부여 |
| Logit Adjustment (τ=1) | 사후 보정 | 학습된 logit에서 사전확률 편향 제거 |

### 3.3 비대칭 오분류 비용 지표

> Cost = (C_FN · FN + C_FP · FP) / N

본 연구는 도메인 가정으로 **C_FN / C_FP = 10 / 1**을 채택하였다.

### 3.4 평가 지표

Recall, F2-Score, G-Mean, PR-AUC, 비대칭 오분류 비용(Cost)

---

## 4. Experimental Setup

- **분할 전략**: Stratified Train(80%) / Test(20%) 분할 + Train 80% 위 Stratified 5-Fold 교차검증
- **비교 모델**:
  - Elastic Net Logistic Regression (선형 판별)
  - Random Forest (배깅 기반 앙상블)
  - CatBoost (부스팅 기반 앙상블)
- **하이퍼파라미터**:
  - Elastic Net: `l1_ratio ∈ {0.2, 0.5, 0.8}`, `C ∈ {0.01, 0.1, 1, 10}` GridSearch
  - Random Forest: `n_estimators=400`
  - CatBoost: `iterations=400, depth=6, learning_rate=0.05`

---

## 5. Main Results

### 5.1 Holdout(Test 20%) 성능 비교

| Model | Variant | Recall | F2 | G-Mean | PR-AUC | **Cost** |
|---|---|---:|---:|---:|---:|---:|
| Elastic Net LR | Baseline | 0.0000 | 0.0000 | 0.0000 | 0.1827 | 0.5999 |
| Elastic Net LR | Class Weighted | 0.9351 | 0.4125 | 0.7438 | 0.1623 | 0.4228 |
| Elastic Net LR | LogitAdj (τ=1) | 0.9297 | 0.4097 | 0.7410 | 0.1827 | 0.4270 |
| Random Forest | Baseline | 0.0108 | 0.0135 | 0.1040 | 0.2453 | 0.5934 |
| Random Forest | Class Weighted | 0.0054 | 0.0067 | 0.0735 | 0.2464 | 0.5966 |
| Random Forest | LogitAdj (τ=1) | 0.9081 | 0.4138 | 0.7461 | 0.2453 | 0.4189 |
| CatBoost | Baseline | 0.0378 | 0.0467 | 0.1944 | 0.2753 | 0.5781 |
| CatBoost | Class Weighted | 0.6595 | 0.4580 | 0.7433 | 0.2506 | 0.3567 |
| **CatBoost** | **LogitAdj (τ=1)** | **0.8703** | **0.4792** | **0.7978** | **0.2753** | **0.3304** |

### 5.2 5-Fold 교차검증 (CatBoost 기준)

| Variant | μ(Cost) | σ(Cost) | μ(G-Mean) |
|---|---:|---:|---:|
| Baseline | 0.5727 | 0.0068 | 0.2098 |
| Class Weighted | 0.3814 | 0.0149 | 0.7116 |
| **LogitAdj (τ=1)** | **0.3338** | **0.0082** | **0.7952** |

### 5.3 핵심 결과 요약

1. **CatBoost + Logit Adjustment** 조합이 Holdout 비용 지표 **0.3304**로 최저값 달성
   - Baseline(0.5781) 대비 **42.9% 감소**
   - Class Weighted(0.3567) 대비 **7.4% 감소**

2. **일반화 안정성**: LogitAdj의 σ(Cost) = 0.0082는 Class Weighted의 σ = 0.0149의 약 절반 수준

3. **모델 무관 일관성**: LogitAdj는 LR, RF, CatBoost 세 모델 모두에서 Recall 87~93%로 일관된 성능. 반면 Class Weighted는 Random Forest에서 Recall 0.54%로 실패

4. **PR-AUC 보존**: Baseline과 LogitAdj의 PR-AUC가 모든 모델에서 동일 (단조변환 성질)

### 5.4 최종 모델 Confusion Matrix

CatBoost + LogitAdj(τ=1), threshold = 0.50

|  | Pred Normal | Pred Fraud |
|---|---:|---:|
| **True Normal** | 2,120 | 779 |
| **True Fraud** | 24 | **161** |

사기 185건 중 **161건 검출 (Recall 87.0%)**

---

## 6. Feature Importance

CatBoost 내장 Feature Importance Top 5 (Logit Adjustment 적용 모델):

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Fault | 30.11 |
| 2 | Year | 10.40 |
| 3 | BasePolicy_Liability | 5.24 |
| 4 | Month_cos | 4.53 |
| 5 | MonthClaimed_cos | 3.67 |

**도메인 검증**: `Fault = Policy Holder`의 실제 사기율 7.89% vs `Third Party` 0.88% — 약 **8.9배** 차이. 모델이 학습한 패턴이 도메인 지식과 일치함을 확인.

---

## 7. Conclusion

본 연구의 결과는 **Logit Adjustment가 복잡한 데이터 전처리나 모델 구조 변경 없이 사전확률 보정만으로 적용 가능**하다는 점에서, 실제 보험사 탐지 시스템에 도입 가능한 실용적 방법론임을 시사한다. 특히 세 가지 이질적인 모델 패러다임(선형, 배깅, 부스팅) 모두에서 일관된 성능 개선이 관찰된 것은, 이 방법의 범용성을 뒷받침하는 실증적 근거가 된다.

### Limitations & Future Work

1. **비용 비율의 도메인 가정**: C_FN/C_FP = 10/1은 도메인 직관에 기반한 가정이며 실제 운영 비용에서 도출된 값이 아니다. 민감도 분석 필요.
2. **τ 튜닝 미반영**: τ = 1을 기본값으로 사용. 모델 캘리브레이션 품질에 따른 최적 τ 탐색이 향후 과제.
3. **시계열 구조 미반영**: i.i.d. Stratified 분할을 사용. Year 변수가 Top 2 중요도를 차지한 것은 시간적 편향 가능성을 시사. 시계열 기반 교차검증 필요.
4. **단일 데이터셋 기반 실험**: 다른 보험 데이터셋이나 실제 운영 환경에서의 외부 검증이 향후 과제.

---

## References

- Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of k-fold cross-validation. *Journal of Machine Learning Research*, 5, 1089–1105.
- Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Krawczyk, B. (2016). Learning from imbalanced data: open challenges and future directions. *Progress in Artificial Intelligence*, 5(4), 221–232.
- Menon, A.K., Jayasumana, S., Rawat, A.S., Jain, H., Veit, A., & Kumar, S. (2021). Long-tail learning via logit adjustment. *ICLR*.
- Mitchell, T. (1997). *Machine Learning*. McGraw-Hill.
- Ngai, E.W.T., Hu, Y., Wong, Y.H., Chen, Y., & Sun, X. (2011). The application of data mining techniques in financial fraud detection: A classification framework and an academic review of literature. *Decision Support Systems*, 50(3), 559–569.
- Salem, O. (2017). The impact of false negative cost on the performance of cost-sensitive learning: A case study in detecting fraudulent transactions. *International Journal of Intelligent Systems and Applications*.
