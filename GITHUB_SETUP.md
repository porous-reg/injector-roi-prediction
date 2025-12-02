# GitHub 저장소 설정 가이드

GitHub에 프로젝트를 업로드하기 위한 단계별 가이드입니다.

## 📝 사전 준비

1. **GitHub 계정 생성** (없는 경우)
   - https://github.com/join

2. **Git 설치** (없는 경우)
   - https://git-scm.com/download/win (Windows)
   - 또는 GitHub Desktop 사용 가능

## 🔧 Windows에서 Git 설정하기

### 방법 1: Git 명령어 사용 (고급)

#### 1단계: Git 초기화

```powershell
# 프로젝트 폴더로 이동
cd C:\Study\ARAMCO\1124Showing_Filter

# Git 저장소 초기화
git init
```

#### 2단계: 파일 추가

```powershell
# 모든 파일 추가
git add .

# 또는 특정 파일만 추가
git add *.py
git add requirements.txt
git add README.md
git add *.pth
git add *.pkl
```

#### 3단계: 첫 커밋

```powershell
git commit -m "Initial commit: Injector ROI Prediction App"
```

#### 4단계: GitHub 저장소 생성 및 연결

1. **GitHub에서 저장소 생성**
   - https://github.com/new 접속
   - Repository name: `injector-roi-prediction`
   - Public 선택
   - "Create repository" 클릭

2. **로컬과 GitHub 연결**

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/injector-roi-prediction.git
```

(⚠️ YOUR_USERNAME을 실제 GitHub 사용자명으로 변경)

#### 5단계: 업로드

```powershell
git push -u origin main
```

**인증 정보 입력:**
- Username: GitHub 사용자명
- Password: Personal Access Token (일반 비밀번호 아님!)
  - 토큰 생성: https://github.com/settings/tokens
  - `repo` 권한 체크
  - "Generate token" 클릭 후 복사

### 방법 2: GitHub Desktop 사용 (초보자 권장)

1. **GitHub Desktop 설치**
   - https://desktop.github.com/

2. **GitHub Desktop 실행**
   - "Sign in to GitHub" 클릭
   - GitHub 계정으로 로그인

3. **저장소 생성**
   - File → New Repository
   - Name: `injector-roi-prediction`
   - Local path: `C:\Study\ARAMCO\1124Showing_Filter`
   - "Create repository" 클릭

4. **파일 추가 및 커밋**
   - 왼쪽에서 변경된 파일 확인
   - Summary에 "Initial commit" 입력
   - "Commit to main" 클릭

5. **GitHub에 푸시**
   - "Publish repository" 클릭
   - Public 선택
   - "Publish repository" 클릭

### 방법 3: GitHub 웹 인터페이스 (가장 간단)

1. **GitHub에서 저장소 생성**
   - https://github.com/new
   - Repository name: `injector-roi-prediction`
   - Public 선택
   - "Create repository" 클릭

2. **파일 업로드**
   - 저장소 페이지에서 "uploading an existing file" 클릭
   - 모든 파일 드래그 앤 드롭:
     - `app*.py` (모든 앱 파일)
     - `requirements.txt`
     - `README.md`
     - `*.pth` (모델 파일)
     - `*.pkl` (스케일러 파일)
   - "Commit changes" 클릭

⚠️ **주의**: 대용량 파일(.pth, .pkl)은 웹에서 업로드할 수 없을 수 있습니다.
- 100MB 이상: Git LFS 사용 또는 다른 방법 사용

## 🔍 대용량 파일 처리

### Git LFS 사용 (권장)

```powershell
# Git LFS 설치
# https://git-lfs.github.com/ 다운로드 및 설치

# Git LFS 초기화
git lfs install

# 대용량 파일 추적
git lfs track "*.pth"
git lfs track "*.pkl"

# 설정 파일 커밋
git add .gitattributes
git commit -m "Add Git LFS tracking"

# 파일 추가
git add *.pth *.pkl
git commit -m "Add model files with LFS"
git push
```

## ✅ 확인 사항

업로드 후 확인:

- [ ] GitHub 저장소에 모든 파일이 표시되는가?
- [ ] 모델 파일(.pth, .pkl)이 포함되어 있는가?
- [ ] `requirements.txt`가 있는가?
- [ ] `README.md`가 있는가?
- [ ] 저장소가 Public으로 설정되어 있는가?

## 🚀 다음 단계

GitHub 업로드 완료 후:
1. [QUICK_START.md](QUICK_START.md) 참조
2. Streamlit Cloud 배포 진행

## 🆘 문제 해결

### "Large files detected" 오류
- Git LFS 사용
- 또는 파일 크기 확인 후 필요시 압축

### "Permission denied" 오류
- Personal Access Token 확인
- 저장소 권한 확인

### "Remote origin already exists" 오류
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/injector-roi-prediction.git
```

## 📚 추가 자료

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git LFS 문서](https://git-lfs.github.com/)

