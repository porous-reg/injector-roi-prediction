# 배포 가이드 (Deployment Guide)

이 문서는 Injector ROI Prediction 앱을 GitHub와 Streamlit Cloud를 통해 온라인에 배포하는 상세 가이드를 제공합니다.

## 📋 사전 준비 사항

1. **GitHub 계정**
   - https://github.com 에서 계정 생성 (없는 경우)

2. **Streamlit Cloud 계정**
   - GitHub 계정으로 자동 로그인 가능

3. **Git 설치** (로컬에서 작업하는 경우)
   - https://git-scm.com/downloads

## 🚀 단계별 배포 가이드

### Step 1: GitHub 저장소 생성

1. **GitHub에 접속하여 새 저장소 생성**
   ```
   https://github.com/new
   ```
   - Repository name: `injector-roi-prediction` (또는 원하는 이름)
   - Description: "AI-based Injector Rate of Injection Prediction System"
   - Public/Private 선택 (Public 권장 - Streamlit Cloud 무료 사용)
   - README.md, .gitignore는 자동 생성하지 않음 (이미 있음)

2. **"Create repository" 클릭**

### Step 2: 로컬 프로젝트를 GitHub에 업로드

#### Windows에서 Git 사용하기

1. **프로젝트 폴더에서 Git 초기화**
   ```powershell
   cd C:\Study\ARAMCO\1124Showing_Filter
   git init
   ```

2. **모든 파일 추가** (모델 파일 포함)
   ```powershell
   git add .
   ```

3. **첫 커밋 생성**
   ```powershell
   git commit -m "Initial commit: Injector ROI Prediction App with models"
   ```

4. **GitHub 저장소 연결**
   ```powershell
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/injector-roi-prediction.git
   ```
   (YOUR_USERNAME을 실제 GitHub 사용자명으로 변경)

5. **업로드**
   ```powershell
   git push -u origin main
   ```
   - GitHub 사용자명과 Personal Access Token 입력 필요
   - Token 생성: https://github.com/settings/tokens

#### 중요: 대용량 파일 처리

모델 파일(.pth, .pkl)이 100MB 이상인 경우:

**옵션 1: Git LFS 사용** (권장)
```powershell
# Git LFS 설치 후
git lfs install
git lfs track "*.pth"
git lfs track "*.pkl"
git add .gitattributes
git add *.pth *.pkl
git commit -m "Add model files with LFS"
git push
```

**옵션 2: GitHub Releases 사용**
- 모델 파일을 ZIP으로 압축
- GitHub Releases에 업로드
- 앱에서 다운로드하도록 코드 수정

**옵션 3: 클라우드 스토리지 사용**
- Google Drive, Dropbox 등에 모델 파일 업로드
- 공유 링크 생성
- 앱에서 URL로 다운로드하도록 수정

### Step 3: Streamlit Cloud 배포

1. **Streamlit Cloud 접속**
   ```
   https://share.streamlit.io/
   ```
   - "Sign in with GitHub" 클릭
   - GitHub 계정으로 로그인

2. **앱 배포 시작**
   - "New app" 버튼 클릭

3. **배포 설정 입력**
   - **Repository**: `YOUR_USERNAME/injector-roi-prediction` 선택
   - **Branch**: `main` 선택
   - **Main file path**: `app.py` 입력
   - **App URL**: 자동 생성되거나 커스텀 가능
     - 예: `injector-lstm` → https://injector-lstm.streamlit.app

4. **"Deploy!" 클릭**
   - 첫 배포는 5-10분 소요
   - 배포 중 로그 확인 가능

5. **배포 완료**
   - 성공하면 공유 가능한 URL 생성
   - 예: `https://injector-lstm.streamlit.app`

### Step 4: 추가 앱 배포 (선택사항)

같은 저장소에서 여러 앱을 배포하려면:

1. Streamlit Cloud 대시보드에서 "New app" 클릭
2. 같은 Repository 선택
3. Main file path만 변경:
   - `app_trm.py` → TRM 모델 앱
   - `app_hrm.py` → HRM 모델 앱
   - `app_inverse_control.py` → Inverse Control 앱

각각 다른 URL을 가집니다.

## 🔧 배포 후 설정

### 환경 변수 설정 (필요한 경우)

1. Streamlit Cloud 앱 페이지에서 "☰" (햄버거 메뉴) 클릭
2. "Settings" 선택
3. "Secrets" 탭에서 환경 변수 추가 가능

예시:
```toml
[secrets]
API_KEY = "your-api-key"
```

### 자동 업데이트

- GitHub에 푸시하면 자동으로 재배포됩니다
- 배포 상태는 Streamlit Cloud 대시보드에서 확인 가능

## 🐛 문제 해결

### 문제 1: "Module not found" 오류

**해결:**
- `requirements.txt`에 누락된 패키지 추가
- GitHub에 커밋 & 푸시
- Streamlit Cloud에서 자동 재배포

### 문제 2: "File not found" 오류 (모델 파일)

**해결:**
- 모델 파일이 저장소에 포함되어 있는지 확인
- 파일 경로가 올바른지 확인
- `.gitignore`에서 파일이 제외되지 않았는지 확인

### 문제 3: 배포가 계속 실패

**해결:**
1. Streamlit Cloud 로그 확인 (배포 페이지에서 "Manage app" → "Logs")
2. 로컬에서 테스트:
   ```bash
   streamlit run app.py
   ```
3. `requirements.txt` 버전 충돌 확인
4. 메모리 부족 문제인 경우 모델 최적화 고려

### 문제 4: 대용량 파일 업로드 실패

**해결:**
- Git LFS 사용
- 또는 클라우드 스토리지 활용
- 또는 모델 파일을 작게 분할

## 📊 모니터링

### Streamlit Cloud 대시보드

- 앱 사용량 통계
- 에러 로그
- 재배포 이력

### 앱 성능 최적화

1. **모델 캐싱**
   - `@st.cache_resource` 데코레이터 사용 (이미 구현됨)

2. **리소스 관리**
   - 불필요한 모델 로드 방지
   - 메모리 효율적인 데이터 처리

## 🔐 보안 고려사항

1. **민감한 정보 보호**
   - API 키, 비밀번호는 Secrets 사용
   - 코드에 하드코딩 금지

2. **공개/비공개 설정**
   - Public 저장소: 모든 사용자 접근 가능
   - Private 저장소: Streamlit Cloud Pro 필요

## 📚 추가 리소스

- [Streamlit Cloud 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [Git LFS 문서](https://git-lfs.github.com/)
- [GitHub 문서](https://docs.github.com/)

## 🎉 완료!

배포가 완료되면 공유 가능한 URL을 통해 전 세계 사용자들이 앱을 사용할 수 있습니다!

