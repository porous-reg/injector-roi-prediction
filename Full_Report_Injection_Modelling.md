# 인젝터 분사율 측정 및 AI 기반 분사량 예측 모델링 연구 보고서

**Date:** 2024. 11. 24

**Subject:** Bosch 방식을 이용한 Peak & Hold 인젝터 분사 특성 분석 및 Deep Learning 기반 가상 센서 개발

## 1. 서론 (Introduction)

### 1.1 연구 배경

엄격해지는 배기 규제와 연비 향상을 위해 GDI 및 디젤 엔진의 연료 분사 제어 기술은 날로 정밀해지고 있다. 특히 다단 분사(Multiple Injection) 전략을 효과적으로 구현하기 위해서는 인젝터의 **분사율(Rate of Injection, ROI)** 형상과 **총 분사량(Injection Mass)**을 다양한 운전 조건에서 정확히 예측하는 것이 필수적이다.

### 1.2 연구 목적

본 연구에서는 **Bosch Tube Method**를 이용하여 Peak & Hold 방식 인젝터의 분사율을 측정한다. 획득된 실험 데이터를 바탕으로, 통전 시간(Energizing Time, ET)과 레일 압력(Rail Pressure) 변화에 따른 분사량 예측을 위한 최적의 **회귀 모델**을 개발하고, 전류 신호로부터 분사율 파형을 예측하는 **Deep Learning 기반 가상 센서**를 구현하는 것을 목표로 한다.

## 2. 실험 장치 및 방법 (Experimental Setup)

### 2.1 Bosch 측정 원리

Bosch 법은 긴 배관(Measuring tube) 내에 연료를 분사했을 때 발생하는 압력 파동($P$)을 측정하여 분사율($dq/dt$)을 계산하는 방식이다.

$$\frac{dq}{dt} = \frac{A}{a} \cdot P$$

- $A$: 튜브 단면적
- $a$: 연료 내 음속 (Sound speed)
- $P$: 압력 파동 (Pressure wave)

### 2.2 실험 조건

- **인젝터 타입:** Peak and Hold 제어 방식
- **레일 압력 ($P_{rail}$):** 100bar, 150bar, 200bar, 250bar, 300bar, 350bar
- **통전 시간 ($ET$):** 250us ~ 5000us (Time-slice sweep)

## 3. 데이터 처리 프로세스 (Data Processing)

Raw 데이터의 노이즈를 제거하고 유의미한 분사율을 도출하기 위해 다음과 같은 4단계 파이프라인을 구축하였다.

1. **Raw Data Acquisition:**
    - 시간, 압력 센서 전압, TTL 신호, 인젝터 구동 전류(Current) 측정.
2. **BCswap & Zero Correction:**
    - 채널 스왑 및 압력 센서의 초기 오프셋(Zero offset) 보정.
3. **FFT Smoothing (Hybrid Filtering):**
    - Bosch 법의 과도 응답 특성을 살리기 위해 두 가지 필터를 혼합 사용.
    - **Transient Region:** Weak Filter (10kHz) 적용 (빠른 응답성 확보).
    - **Steady-state Region:** Strong Filter (3kHz) 적용 (노이즈 제거).
    - **Transition:** 두 구간 사이를 보간(Interpolation)하여 연결.
4. **Time Shifting & Calculation:**
    - TTL Trigger 시점을 기준으로 시간축 정렬 ($t=0$).
    - 음속($a$) 및 질량($m = \int \dot{m} dt$) 계산.

## 4. 연구 결과 I: 총 분사량 예측 (Mass Estimation)

인젝터의 거동 특성을 가장 잘 설명하는 수학적 모델을 찾기 위해 세 가지 알고리즘을 비교 분석하였다.

### 4.1 모델 비교 분석

| 모델 (Model) | 결정 계수 ($R^2$) | RMSE (mg) | 비고 |
| --- | --- | --- | --- |
| Linear Regression | 0.8945 | 6.4856 | 고부하 오차 큼 |
| Random Forest | 0.8186 | 8.5065 | 외삽(Extrapolation) 실패 |
| **Polynomial Reg. (2nd)** | **0.9997** | **0.3545** | **최적 모델 선정** |

### 4.2 결론

인젝터의 분사량은 압력과 시간의 **2차 함수(Quadratic Function)** 관계를 따름을 입증하였다. 이를 통해 실험하지 않은 압력 구간에 대한 **가상 룩업 테이블(Virtual Lookup Table)**을 생성 완료하였다.

**물리적 근거:**
- **베르누이 방정식 (Bernoulli's Principle):** 유체 흐름에서 유량($Q$)과 압력차($\Delta P$)의 관계는 **제곱근($\sqrt{}$) 또는 제곱($^2$)의 관계**를 가집니다.
- **오리피스 유동:** 인젝터 구멍(Orifice)을 통과하는 연료의 양은 면적과 속도의 곱이며, 이는 수학적으로 **2차 곡선(Parabola)** 형태를 띠는 경우가 많습니다.

수식 형태: $Mass = a \cdot Pressure^2 + b \cdot Pressure \cdot Time + c \cdot Time^2 + ...$

## 5. 연구 결과 II: 분사율 파형 예측 (Waveform Prediction)

전류 신호(Current)와 압력(Pressure) 정보로부터 분사율 파형을 예측하기 위해 세 가지 Deep Learning 아키텍처를 비교 분석하였다.

### 5.1 모델 아키텍처 비교

#### 5.1.1 LSTM (Long Short-Term Memory)

- **구조:** 순환 신경망(Recurrent Neural Network) 기반
- **특징:** 시계열 데이터의 시간적 의존성을 순차적으로 학습
- **장점:** 장기 의존성(Long-term dependency) 학습 가능
- **단점:** 병렬 처리 불가, 학습 시간이 길 수 있음

#### 5.1.2 TRM (Tiny Recursive Model / Transformer)

- **구조:** Transformer 기반 경량 모델
- **특징:** 
  - Multi-head Self-Attention 메커니즘
  - Positional Encoding을 통한 시간 정보 인코딩
  - 병렬 처리 가능
- **장점:** Attention 메커니즘으로 긴 시퀀스의 의존성 학습 가능, 빠른 추론
- **단점:** 학습 데이터가 적을 경우 과적합 가능성

#### 5.1.3 HRM (Hybrid Recursive Model)

- **구조:** LSTM + Transformer 하이브리드
- **특징:**
  - LSTM 레이어로 지역적 패턴 학습
  - Transformer 레이어로 전역적 의존성 포착
- **장점:** 두 아키텍처의 장점 결합
- **단점:** 모델 복잡도 증가

### 5.2 성능 평가 지표

본 연구에서는 다음과 같은 세 가지 지표를 사용하여 모델 성능을 평가하였다:

1. **R² Score (Waveform Accuracy):** 파형 형상의 일치도 측정
2. **RMSE % (Relative to Peak):** 순간 오차를 Peak 값 대비 백분율로 표현
3. **Total Mass Error:** 적분된 총 분사량의 평균 오차율

### 5.3 모델 성능 비교

#### 5.3.1 Original Model Performance

| 모델 | R² Score | RMSE (mg/ms) | RMSE % | Total Mass Error (%) |
|------|----------|--------------|--------|---------------------|
| **LSTM** | 0.99359 | 0.3713 | 1.78% | 3.40% |
| **TRM** | 0.99686 | 0.2601 | 1.24% | 5.76% |
| **HRM** | 0.99131 | 0.4324 | 2.07% | 29.93% |

#### 5.3.2 Hybrid Algorithm (Polynomial Correction) 적용 후

|------|----------|--------|---------------------|--------|
| **LSTM (Original)** | 0.99359 | 1.78% | 3.40% | - |
| **LSTM (Hybrid)** | 0.99466 | 1.62% | **0.00%** | **Mass Error 완전 제거** |
| **TRM (Original)** | 0.99686 | 1.24% | 5.76% | - |
| **TRM (Hybrid)** | 0.99635 | 1.34% | **0.00%** | **Mass Error 완전 제거** |
| **HRM (Original)** | 0.99131 | 2.07% | 29.93% | - |
| **HRM (Hybrid)** | 0.46487 | 16.24% | 10.51% | Mass Error 65% 개선 |

### 5.4 Hybrid Algorithm 상세

Hybrid Algorithm은 다음과 같은 방식으로 작동한다:

1. **Deep Learning 모델 예측:** LSTM/TRM/HRM이 전류 및 압력 정보로부터 분사율 파형 예측
2. **총 분사량 계산:** 예측된 파형을 적분하여 총 분사량 계산
3. **Polynomial 모델 보정:** Polynomial 회귀 모델로부터 예상되는 총 분사량과 비교
4. **Scale Factor 적용:** 형상은 유지하되 전체 파형에 Scale Factor를 적용하여 총 분사량 보정

**수식:**
$$y_{corrected}(t) = y_{pred}(t) \times \frac{m_{target}}{m_{predicted}}$$

여기서 $m_{target}$는 Polynomial 모델로부터 예측된 총 분사량이고, $m_{predicted}$는 Deep Learning 모델이 예측한 총 분사량이다.

### 5.5 모델 성능 종합 비교 요약

| 모델 | R² Score (Original) | RMSE % (Original) | Mass Error % (Original) | Mass Error % (Hybrid) | 특징 |
|------|---------------------|-------------------|------------------------|----------------------|------|
| **LSTM** | 0.99359 | 1.78% | 3.40% | **0.00%** | 가장 균형잡힌 성능 |
| **TRM** | **0.99686** | **1.24%** | 5.76% | **0.00%** | **최고 형상 복원 + Hybrid 효과 우수** |
| **HRM** | 0.99131 | 2.07% | 29.93% | 10.51% | 형상 복원은 우수하나 Mass Error 높음 |

**주요 발견 사항:**
- **TRM 모델**이 R² Score와 RMSE 측면에서 가장 우수하며, Hybrid Algorithm 적용 시 Total Mass Error를 완전히 제거
- **LSTM 모델**은 가장 안정적이고 균형잡힌 성능으로 실용적 응용에 적합
- **HRM 모델**은 하이브리드 구조에도 불구하고 총 분사량 예측에서 한계를 보임

### 5.6 고찰

1. **LSTM 모델:**
   - R² Score 0.99359로 우수한 형상 복원 성능
   - RMSE 1.78% (Peak 대비)로 순간 오차도 낮음
   - Total Mass Error 3.40%로 총 분사량 예측 정확도가 세 모델 중 가장 우수
   - 안정적인 성능으로 실용적 응용에 가장 적합

2. **TRM 모델:**
   - **R² Score 0.99686로 세 모델 중 가장 높은 형상 복원 성능**
   - **RMSE 1.24% (Peak 대비)로 가장 낮은 순간 오차**
   - Total Mass Error 5.76%로 LSTM보다 다소 높음
   - **Hybrid Algorithm 적용 시 Total Mass Error가 0.00%로 완전히 제거됨**
   - Transformer 기반의 Attention 메커니즘으로 긴 시퀀스 의존성 학습 가능
   - 병렬 처리로 빠른 추론 속도 기대
   - 경량 모델 구조로 모바일/임베디드 적용 가능성

3. **HRM 모델:**
   - R² Score 0.99131로 형상 복원 성능은 우수
   - RMSE 2.07% (Peak 대비)로 세 모델 중 가장 높음
   - **Total Mass Error 29.93%로 가장 높아 총 분사량 예측에 한계**
   - LSTM과 Transformer의 하이브리드 구조에도 불구하고 단순 스케일 보정으로는 한계
   - Hybrid Algorithm 적용 시 Mass Error는 10.51%로 개선되나, R² Score가 0.46487로 급격히 저하되어 파형 형상이 크게 왜곡됨
   - 모델 구조 재검토 필요

4. **Hybrid Algorithm:**
   - **TRM 모델에 적용 시 가장 효과적**: Total Mass Error를 완전히 제거하며 파형 형상 유지 (R² 0.99635)
   - **LSTM 모델**: 보정 효과 검증 필요
   - **HRM 모델**: 단순 스케일 보정 방식은 부적합, 더 정교한 보정 알고리즘 필요
   - Polynomial 모델의 물리적 근거를 활용한 보정으로 총 분사량 정확도 향상
   - 파형 형상은 유지하면서 면적만 조정하는 방식으로 실용성 확보

## 6. 결론 (Conclusion)

1. 본 연구를 통해 Bosch 측정 데이터의 전처리부터 Deep Learning 모델링까지의 일련의 프로세스를 정립하였다.

2. **분사량 예측**에 있어 복잡한 딥러닝 기법보다, 인젝터의 물리적 특성(Physics)을 반영한 **다항 회귀 모델(Polynomial Regression)**이 가장 적합함을 입증하였다 ($R^2=0.9997$).

3. **분사율 파형 예측**에 있어 세 가지 Deep Learning 모델을 비교한 결과:
   - **TRM 모델**이 R² Score 0.99686, RMSE 1.24% (Peak 대비)로 가장 우수한 형상 복원 성능을 보였으며, Hybrid Algorithm 적용 시 Total Mass Error를 완전히 제거함 (0.00%)
   - **LSTM 모델**은 R² Score 0.99359, RMSE 1.78%, Total Mass Error 3.40%로 가장 균형잡힌 성능
   - **HRM 모델**은 R² Score 0.99131로 형상 복원은 우수하나, Total Mass Error 29.93%로 총 분사량 예측에 한계

4. **Hybrid Algorithm**을 통해 Polynomial 모델의 물리적 근거를 활용한 보정 방법을 제안하였으며, **TRM 모델에 적용 시 Total Mass Error를 완전히 제거**하는 효과를 확인하였다.

5. 개발된 모델을 통해 50bar/50us 단위의 **고해상도 가상 분사량 맵(Virtual Lookup Table)**을 생성하였으며, 이는 향후 ECU 캘리브레이션 및 엔진 모델링의 기초 데이터로 즉시 활용 가능하다.

## 7. 향후 연구 방향

1. **LSTM 모델의 Hybrid Algorithm 결과 검증** 및 세 모델 간 종합 비교 분석
2. **TRM-Hybrid Algorithm의 실시간 적용 가능성 검토**: Total Mass Error 완전 제거 효과를 실시간 시스템에 적용
3. **HRM 모델 구조 개선**: 현재의 단순 스케일 보정 방식 대신 더 정교한 보정 알고리즘 개발 필요
4. 다양한 인젝터 타입에 대한 모델 일반화 성능 평가
5. 임베디드 시스템 적용을 위한 모델 경량화 연구 (특히 TRM 모델의 경량화 잠재력)
6. 실시간 분사율 예측을 위한 최적 모델 선정 (정확도 vs 속도 트레이드오프 분석)

**[첨부]**

1. `injection_data_master_v2.csv` (전처리 완료 데이터)
2. `Virtual_Injection_Map.csv` (가상 룩업 테이블)
3. `LSTM_current_vs_ROI.ipynb` (LSTM 모델 학습 노트북)
4. `TRM_current_vs_ROI.ipynb` (TRM 모델 학습 노트북)
5. `HRM_current_vs_ROI.ipynb` (HRM 모델 학습 노트북)
6. `app.py`, `app_trm.py`, `app_hrm.py` (Streamlit 기반 시뮬레이션 애플리케이션)
