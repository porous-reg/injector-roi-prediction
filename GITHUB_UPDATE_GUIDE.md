# GitHub 업데이트 가이드

## 현재 상태
- ✅ 일부 파일이 이미 staged 상태 (커밋 준비 완료)
- ⚠️ 일부 파일이 아직 unstaged 상태

## GitHub에 반영하는 방법

### 방법 1: 모든 변경사항을 한 번에 추가 (권장)

```bash
# 1. 모든 변경사항 추가 (새 파일 + 수정된 파일)
git add .

# 2. 변경사항 확인
git status

# 3. 커밋 (메시지 작성)
git commit -m "Update: LSTM을 DNN으로 변경하고 ET만 입력받도록 수정

- 전류 파형 예측 모델을 LSTM에서 DNN (3 layers)로 변경
- Input을 Pressure+ET에서 ET만 받도록 수정
- 모든 레이어 사이에 GELU activation function 적용
- 모든 앱 파일들(app.py, app_trm.py, app_hrm.py 등) 업데이트
- 데이터 추출 스크립트 업데이트 (ET만 추출)
- 노트북 업데이트 (LSTM_ET_to_Current.ipynb)"

# 4. GitHub에 푸시
git push origin main
```

### 방법 2: 단계별로 선택하여 추가

```bash
# 1. 특정 파일만 추가
git add LSTM_ET_to_Current.ipynb
git add LSTM_current_vs_ROI.ipynb
git add data_analysis_modeling.ipynb

# 2. 변경사항 확인
git status

# 3. 커밋
git commit -m "Update notebooks with DNN architecture"

# 4. 푸시
git push origin main
```

## 주의사항

### 큰 파일 처리
현재 `.gitignore`에서 데이터 파일들이 주석 처리되어 있습니다:
- `*.npz` (데이터셋 파일)
- `*.pkl` (스케일러 파일)
- `*.pth` (모델 파일)

이러한 파일들은 크기가 클 수 있으므로:
1. **작은 파일 (1MB 미만)**: 커밋 가능
2. **큰 파일 (1MB 이상)**: Git LFS 사용 또는 제외 권장

### .gitignore 수정 (선택사항)
큰 모델/데이터 파일을 제외하려면 `.gitignore`를 다음과 같이 수정:

```gitignore
# Data files
*.npz
*.pkl
*.pth
current_waveform_dataset.npz
current_waveform_model.pth
current_scaler_*.pkl
```

## 단계별 실행

### 1단계: 변경사항 추가
```bash
git add .
```

### 2단계: 커밋 메시지와 함께 커밋
```bash
git commit -m "Update: LSTM to DNN architecture and ET-only input

- Changed current waveform predictor from LSTM to DNN (3 layers)
- Updated input from (Pressure, ET) to ET only
- Added GELU activation functions between all layers
- Updated all app files and data extraction scripts"
```

### 3단계: GitHub에 푸시
```bash
git push origin main
```

## 문제 해결

### 원격 저장소와 충돌이 발생하는 경우
```bash
# 1. 최신 변경사항 가져오기
git pull origin main

# 2. 충돌 해결 후 다시 커밋
git add .
git commit -m "Resolve conflicts"
git push origin main
```

### 커밋 메시지를 수정하려는 경우
```bash
# 마지막 커밋 메시지 수정
git commit --amend -m "새로운 커밋 메시지"
git push origin main --force  # 주의: force push는 신중하게 사용
```

## 현재 staged 파일 확인
```bash
git status
```

## 변경사항 상세 확인
```bash
git diff  # unstaged 변경사항
git diff --staged  # staged 변경사항
```

