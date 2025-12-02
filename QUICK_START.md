# 빠른 시작 가이드 (Quick Start Guide)

GitHub와 Streamlit Cloud를 통한 빠른 배포 가이드입니다.

## ⚡ 5분 만에 배포하기

### 1단계: GitHub 저장소 생성 (2분)

1. https://github.com/new 접속
2. 저장소 이름 입력: `injector-roi-prediction`
3. Public 선택 (Streamlit Cloud 무료 사용)
4. "Create repository" 클릭

### 2단계: 파일 업로드 (2분)

**방법 A: GitHub 웹 인터페이스 사용** (가장 쉬움)

1. GitHub 저장소 페이지에서 "uploading an existing file" 클릭
2. 모든 파일 드래그 앤 드롭:
   - `app.py`, `app_trm.py`, `app_hrm.py` 등 모든 .py 파일
   - `requirements.txt`
   - `README.md`
   - `lstm_model.pth`, `trm_model.pth`, `hrm_model.pth`
   - `scaler_X.pkl`, `scaler_y.pkl`
   - `final_injector_model.pkl`, `poly_feature_transformer.pkl`
   - 기타 필요한 파일들
3. "Commit changes" 클릭

**방법 B: Git 명령어 사용**

```bash
cd C:\Study\ARAMCO\1124Showing_Filter
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/injector-roi-prediction.git
git push -u origin main
```

### 3단계: Streamlit Cloud 배포 (1분)

1. https://share.streamlit.io/ 접속
2. "Sign in with GitHub" 클릭
3. "New app" 버튼 클릭
4. 설정 입력:
   - Repository: `YOUR_USERNAME/injector-roi-prediction`
   - Branch: `main`
   - Main file path: `app.py`
5. "Deploy!" 클릭
6. 완료! 🎉

## ✅ 체크리스트

배포 전 확인사항:

- [ ] 모든 Python 파일 (`app.py`, `app_trm.py` 등) 포함
- [ ] `requirements.txt` 파일 포함
- [ ] `README.md` 파일 포함
- [ ] 모든 모델 파일 (.pth, .pkl) 포함
- [ ] GitHub 저장소가 Public으로 설정됨
- [ ] Streamlit Cloud에 GitHub 계정 연결됨

## 🔗 앱 URL 확인

배포 완료 후:
- Streamlit Cloud 대시보드에서 앱 URL 확인
- 예: `https://injector-lstm.streamlit.app`

## 📱 추가 앱 배포

같은 저장소에서 다른 앱도 배포하려면:

1. Streamlit Cloud에서 "New app" 클릭
2. 같은 Repository 선택
3. Main file path만 변경:
   - `app_trm.py` → TRM 앱
   - `app_hrm.py` → HRM 앱
   - `app_inverse_control.py` → Inverse Control 앱

## 🆘 문제가 발생했나요?

자세한 내용은 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)를 참조하세요.

