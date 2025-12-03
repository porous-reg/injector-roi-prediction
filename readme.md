# Injector ROI Prediction - AI Digital Twin

Bosch 방법을 이용한 인젝터 분사율(Rate of Injection, ROI) 측정 데이터를 기반으로, Deep Learning 모델을 활용하여 전류 신호로부터 분사율 파형을 예측하는 AI 기반 가상 센서(Virtual Sensor) 시스템입니다.

## 🚀 온라인 데모

### Streamlit Cloud 배포 (추천)
- **LSTM 모델**: [링크 추가 예정]
- **TRM 모델**: [링크 추가 예정]
- **HRM 모델**: [링크 추가 예정]
- **Inverse Control (LSTM)**: [링크 추가 예정]

## 📋 주요 기능

### 1. Forward Control (Forward Simulation)
- 전류 파형 및 압력 정보로부터 분사율 파형 예측
- LSTM, TRM, HRM 세 가지 Deep Learning 모델 지원
- Polynomial 회귀 모델을 통한 Hybrid 보정 알고리즘
- 실시간 시뮬레이션 및 실제 데이터 파일 업로드 지원

### 2. Inverse Control (Inverse Control)
- 목표 분사량 및 압력으로부터 필요한 통전 시간 계산
- Binary Search 기반 최적화 알고리즘
- 전류 파형 자동 생성 및 시각화

## 🛠️ 설치 방법

### 로컬 환경에서 실행

1. **저장소 클론**
```bash
git clone https://github.com/YOUR_USERNAME/injector-roi-prediction.git
cd injector-roi-prediction
```

2. **가상 환경 생성 및 활성화**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **필요한 패키지 설치**
```bash
pip install -r requirements.txt
```

4. **Streamlit 앱 실행**
```bash
# LSTM 모델 사용
streamlit run app.py

# TRM 모델 사용
streamlit run app_trm.py

# HRM 모델 사용
streamlit run app_hrm.py

# Inverse Control (LSTM)
streamlit run app_inverse_control.py
```

## 📦 프로젝트 구조

```
injector-roi-prediction/
├── app.py                      # LSTM Forward Control 앱
├── app_trm.py                  # TRM Forward Control 앱
├── app_hrm.py                  # HRM Forward Control 앱
├── app_inverse_control.py      # LSTM Inverse Control 앱
├── app_inverse_control_trm.py  # TRM Inverse Control 앱
├── app_inverse_control_hrm.py  # HRM Inverse Control 앱
├── requirements.txt            # Python 패키지 의존성
├── README.md                   # 프로젝트 설명서
├── Full_Report_Injection_Modelling.md  # 상세 연구 보고서
│
├── 모델 파일들
├── lstm_model.pth             # LSTM 학습된 모델
├── trm_model.pth              # TRM 학습된 모델
├── hrm_model.pth              # HRM 학습된 모델
├── scaler_X.pkl               # 입력 스케일러
├── scaler_y.pkl               # 출력 스케일러
├── final_injector_model.pkl   # Polynomial 회귀 모델
└── poly_feature_transformer.pkl  # 다항식 특성 변환기
│
├── 데이터 파일들
├── injection_data_master_v2.csv  # 전처리된 실험 데이터
└── Virtual_Injection_Map.csv     # 가상 분사량 맵
│
└── 학습 노트북들
    ├── LSTM_current_vs_ROI.ipynb
    ├── TRM_current_vs_ROI.ipynb
    └── HRM_current_vs_ROI.ipynb
```

## 🌐 Streamlit Cloud 배포 가이드

### 1. GitHub 저장소 준비

1. **GitHub에 새 저장소 생성**
   - GitHub에서 새 저장소를 만듭니다
   - 저장소 이름: `injector-roi-prediction` (또는 원하는 이름)

2. **로컬 파일들을 GitHub에 업로드**
```bash
git init
git add .
git commit -m "Initial commit: Injector ROI Prediction App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/injector-roi-prediction.git
git push -u origin main
```

**중요: 모델 파일(.pth, .pkl)도 함께 업로드해야 합니다!**
- 파일 크기가 100MB 이상인 경우 Git LFS를 사용하거나 GitHub Releases에 업로드하는 것을 고려하세요.

### 2. Streamlit Cloud 배포

1. **Streamlit Cloud 접속**
   - https://share.streamlit.io/ 접속
   - GitHub 계정으로 로그인

2. **앱 배포**
   - "New app" 버튼 클릭
   - **Repository**: `YOUR_USERNAME/injector-roi-prediction`
   - **Branch**: `main`
   - **Main file path**: 
     - `app.py` (LSTM Forward Control)
     - 또는 `app_trm.py`, `app_hrm.py` 등
   - "Deploy!" 버튼 클릭

3. **배포 완료**
   - 몇 분 후 앱이 자동으로 배포됩니다
   - 공유 가능한 URL이 생성됩니다

### 3. 여러 앱 배포하기

Streamlit Cloud에서는 하나의 저장소에 여러 앱을 배포할 수 있습니다:

1. 각 앱에 대해 별도의 배포를 생성
2. Main file path만 변경:
   - `app.py` → LSTM Forward Control
   - `app_trm.py` → TRM Forward Control
   - `app_hrm.py` → HRM Forward Control
   - `app_inverse_control.py` → LSTM Inverse Control
   - 등등...

## 📊 모델 성능

### Original Model Performance

| 모델 | R² Score | RMSE % | Total Mass Error (%) |
|------|----------|--------|---------------------|
| **LSTM** | 0.99359 | 1.78% | 3.40% |
| **TRM** | 0.99686 | 1.24% | 5.76% |
| **HRM** | 0.99131 | 2.07% | 29.93% |

### Hybrid Algorithm (Polynomial Correction)

| 모델 | R² Score | RMSE % | Total Mass Error (%) |
|------|----------|--------|---------------------|
| **LSTM (Hybrid)** | 0.99466 | 1.62% | **0.00%** |
| **TRM (Hybrid)** | 0.99635 | 1.34% | **0.00%** |
| **HRM (Hybrid)** | 0.46487 | 16.24% | 10.51% |

자세한 내용은 [Full_Report_Injection_Modelling.md](Full_Report_Injection_Modelling.md)를 참조하세요.

## 📝 사용법

### Forward Control (분사율 예측)

1. **시뮬레이션 모드**
   - 사이드바에서 "Simulation (Hysteresis)" 선택
   - Rail Pressure와 Energizing Time 조정
   - 자동으로 생성된 전류 파형으로부터 분사율 예측

2. **실제 데이터 모드**
   - 사이드바에서 "Upload Real File (.lvm)" 선택
   - `.lvm` 파일 업로드 (Shifted 형식)
   - 압력 정보는 슬라이더로 입력

### Inverse Control (통전 시간 계산)

1. 목표 분사량(Target Mass) 입력
2. Rail Pressure 설정
3. Binary Search 알고리즘이 자동으로 최적 통전 시간 계산
4. 생성된 전류 파형 및 예측된 분사율 시각화

## 🔧 문제 해결

### 모델 파일이 없다는 오류
- 모든 `.pth` 및 `.pkl` 파일이 저장소에 포함되어 있는지 확인
- 파일 크기가 100MB 이상인 경우 Git LFS 사용 고려

### Streamlit Cloud 배포 실패
- `requirements.txt` 파일이 올바른지 확인
- 로그에서 오류 메시지 확인
- 모델 파일 경로가 올바른지 확인

## 📄 라이센스

[라이센스 정보 추가]

## 👥 기여자

[기여자 정보 추가]

## 📧 문의

[연락처 정보 추가]
