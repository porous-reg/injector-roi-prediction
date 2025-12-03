"""
전류 파형 데이터셋 추출 스크립트
Input: ET (Energizing Time)
Output: Current Waveform (1300 points)

이 스크립트는 LVM 파일들에서 전류 파형을 추출하여
ET → Current Waveform 매핑 데이터셋을 생성합니다.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# ==========================================
# 사용자 설정
# ==========================================
ROOT_DIR = '.' 
FILE_PATTERN = "**/*fftSmooth_shifted.lvm"
OUTPUT_FILENAME = "current_waveform_dataset.npz"

# 시간축 설정 (ROI 모델과 동일하게 맞춤)
TIME_START = -0.0005  # -0.5 ms
TIME_END = 0.0060     # 6.0 ms
TARGET_POINTS = 1300  # 해상도

# ==========================================
# 함수 정의
# ==========================================
def parse_filename_info(filepath):
    """파일 경로에서 Pressure와 ET 정보 추출"""
    path_str = str(filepath)
    
    # Pressure 추출 (100bar, 200bar 등)
    pressure_match = re.search(r'(\d+)\s*bar', path_str, re.IGNORECASE)
    pressure = int(pressure_match.group(1)) if pressure_match else None
    
    # ET 추출 (250us, 1000us 등)
    et_match = re.search(r'(\d+)\s*us', path_str, re.IGNORECASE)
    et = int(et_match.group(1)) if et_match else None
    
    return pressure, et

def process_file(filepath):
    """LVM 파일에서 전류 파형 추출"""
    try:
        # 데이터 로드
        df = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        
        # Shifted 파일 기준: 4열이 Current
        if df.shape[1] < 4:
            return None
        
        time_raw = df.iloc[:, 0].values  # 시간 (Column 1)
        current_raw = df.iloc[:, 3].values  # 전류 (Column 4)
        
        # 데이터 타입 변환
        time_raw = pd.to_numeric(time_raw, errors='coerce')
        current_raw = pd.to_numeric(current_raw, errors='coerce')
        
        # NaN 제거
        mask = ~np.isnan(time_raw) & ~np.isnan(current_raw)
        time_raw = time_raw[mask]
        current_raw = current_raw[mask]
        
        if len(time_raw) < 10:
            return None
        
        # Resampling: 균일한 시간축으로 재샘플링
        new_time = np.linspace(TIME_START, TIME_END, TARGET_POINTS)
        f_current = interp1d(time_raw, current_raw, kind='linear', bounds_error=False, fill_value=0)
        new_current = f_current(new_time)
        
        # 음수 제거 (전류는 0 이상)
        new_current = np.maximum(new_current, 0)
        
        return new_current
        
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return None

def main():
    print("=" * 60)
    print("Current Waveform 데이터셋 추출 시작...")
    print("=" * 60)
    
    # 파일 검색
    files = glob.glob(os.path.join(ROOT_DIR, FILE_PATTERN), recursive=True)
    print(f"발견된 파일 수: {len(files)}\n")
    
    # 데이터 저장용 리스트
    X_meta = []  # Input: (Pressure, ET) 메타데이터
    y_current = []  # Output: Current Waveform
    
    ignored_files = []
    cnt = 0
    
    for filepath in files:
        # Pressure와 ET 추출
        pressure, et = parse_filename_info(filepath)
        
        if et is None:
            ignored_files.append({
                "file": filepath,
                "reason": "ET not found in filename",
                "pressure": pressure,
                "et": et
            })
            continue
        
        # 전류 파형 추출
        current_waveform = process_file(filepath)
        
        if current_waveform is None:
            ignored_files.append({
                "file": filepath,
                "reason": "Failed to extract current waveform",
                "pressure": pressure,
                "et": et
            })
            continue
        
        # 데이터 추가 (ET만 사용)
        X_meta.append(et)  # ET만 저장
        y_current.append(current_waveform)
        cnt += 1
        
        if cnt % 10 == 0:
            print(f"  {cnt}개 파일 처리 완료...")
    
    # NumPy 배열로 변환
    X_meta_array = np.array(X_meta)  # (N,) -> ET only
    y_current_array = np.array(y_current)  # (N, 1300) -> Current Waveform
    
    # 1차원 배열을 2차원으로 변환 (scaler를 위해)
    X_meta_array = X_meta_array.reshape(-1, 1)  # (N, 1)
    
    print(f"\n{'='*60}")
    print(f"[추출 완료]")
    print(f"{'='*60}")
    print(f"[OK] 성공적으로 처리된 샘플: {len(X_meta_array)}개")
    print(f"[SKIP] 무시된 파일: {len(ignored_files)}개")
    print(f"\n데이터 Shape:")
    print(f"  Input (ET): {X_meta_array.shape}")
    print(f"  Output (Current Waveform): {y_current_array.shape}")
    
    # 통계 정보
    print(f"\n통계 정보:")
    print(f"  ET 범위: {X_meta_array[:, 0].min()} ~ {X_meta_array[:, 0].max()} us")
    print(f"  Current Peak 범위: {y_current_array.max(axis=1).min():.2f} ~ {y_current_array.max(axis=1).max():.2f} A")
    
    # 무시된 파일 정보 출력
    if ignored_files:
        print(f"\n{'='*60}")
        print(f"무시된 파일 정보")
        print(f"{'='*60}")
        for item in ignored_files[:5]:  # 처음 5개만 출력
            print(f"  - {os.path.basename(item['file'])}")
            print(f"    이유: {item['reason']}")
        if len(ignored_files) > 5:
            print(f"  ... 외 {len(ignored_files) - 5}개 파일")
    
    # 데이터셋 저장
    np.savez_compressed(
        OUTPUT_FILENAME,
        X_meta=X_meta_array,      # (N, 1) [ET]
        y_current=y_current_array  # (N, 1300) Current Waveform
    )
    
    print(f"\n{'='*60}")
    print(f"[저장 완료]")
    print(f"{'='*60}")
    print(f"데이터셋 파일: {os.path.abspath(OUTPUT_FILENAME)}")
    print(f"\n데이터셋 구조:")
    print(f"  - X_meta: Input metadata (ET only)")
    print(f"  - y_current: Output current waveform (1300 points)")

if __name__ == "__main__":
    main()

