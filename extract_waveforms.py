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
OUTPUT_FILENAME = "waveform_dataset.npz"

# 데이터를 1000개의 포인트로 통일 (Resampling)
# 5000us(5ms) 데이터도 1000개, 250us 데이터도 1000개로 맞춤? 
# -> 아니오, 시간축(Time)을 기준으로 잘라야 물리적 의미가 유지됩니다.
# 설정: -0.5ms 부터 6.0ms 까지 총 6.5ms 구간을 1300개 포인트로 고정 (5us 간격)
TIME_START = -0.0005 # -0.5 ms
TIME_END = 0.0060    # 6.0 ms
TARGET_POINTS = 1300 # 해상도 조절 가능

# ==========================================
# 함수 정의
# ==========================================
def parse_pressure(filepath):
    path_str = str(filepath)
    match = re.search(r'(\d+)\s*bar', path_str, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def process_file(filepath):
    try:
        # 데이터 로드
        df = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        
        # 필요한 컬럼 추출 (Shifted 파일 기준)
        # Col 1: Time (Shifted)
        # Col 4: Current (Injector Input Current) - 원본의 Col 4가 그대로 옴
        # Col 11: Mass Flow Rate (Calculated)
        
        # 데이터가 11열보다 적으면 스킵
        if df.shape[1] < 11:
            return None

        time_raw = df.iloc[:, 0].values
        current_raw = df.iloc[:, 3].values
        mass_raw = df.iloc[:, 10].values # Index 10 -> Column 11

        # 데이터가 문자열로 읽혔을 경우를 대비해 변환
        time_raw = pd.to_numeric(time_raw, errors='coerce')
        current_raw = pd.to_numeric(current_raw, errors='coerce')
        mass_raw = pd.to_numeric(mass_raw, errors='coerce')
        
        # 유효한 숫자 데이터만 남김 (NaN 제거)
        mask = ~np.isnan(time_raw) & ~np.isnan(current_raw) & ~np.isnan(mass_raw)
        time_raw = time_raw[mask]
        current_raw = current_raw[mask]
        mass_raw = mass_raw[mask]

        if len(time_raw) < 10:
            return None

        # --- Resampling (핵심) ---
        # 목표하는 균일한 시간축 생성
        new_time = np.linspace(TIME_START, TIME_END, TARGET_POINTS)
        
        # 보간 함수 생성 (Interpolation)
        # fill_value=0: 측정 범위 밖의 시간은 0으로 채움 (Pre/Post injection 구간)
        f_current = interp1d(time_raw, current_raw, kind='linear', bounds_error=False, fill_value=0)
        f_mass = interp1d(time_raw, mass_raw, kind='linear', bounds_error=False, fill_value=0)
        
        new_current = f_current(new_time)
        new_mass = f_mass(new_time)

        return new_current, new_mass

    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return None

def main():
    print("Waveform 데이터 추출 시작...")
    files = glob.glob(os.path.join(ROOT_DIR, FILE_PATTERN), recursive=True)
    print(f"발견된 파일 수: {len(files)}")

    X_list = [] # Input (Current, Pressure)
    y_list = [] # Output (Mass Flow)
    
    cnt = 0
    for filepath in files:
        pressure = parse_pressure(filepath)
        if pressure == 0: continue
            
        result = process_file(filepath)
        if result is None: continue
        
        current_wave, mass_wave = result
        
        # --- 입력 데이터(X) 구성 ---
        # LSTM은 (TimeSteps, Features) 형태를 받습니다.
        # Feature 1: Current Waveform (시간에 따라 변함)
        # Feature 2: Rail Pressure (시간에 따라 일정함 -> 상수로 채움)
        pressure_wave = np.full_like(current_wave, pressure) 
        
        # 합치기 (Shape: [1300, 2])
        combined_X = np.stack([current_wave, pressure_wave], axis=1)
        
        X_list.append(combined_X)
        y_list.append(mass_wave)
        cnt += 1
        
        if cnt % 10 == 0:
            print(f"{cnt}개 파일 처리 완료...")

    # Numpy 배열로 변환
    X_final = np.array(X_list) # (N, 1300, 2)
    y_final = np.array(y_list) # (N, 1300) -> Output은 보통 1차원 시퀀스

    # 차원 확인
    # X: (Samples, TimeSteps, Features=2)
    # y: (Samples, TimeSteps, Features=1)
    y_final = y_final.reshape(y_final.shape[0], y_final.shape[1], 1)

    print("\n[추출 완료]")
    print(f"Input Shape (X): {X_final.shape} -> (샘플수, 시간스텝, [전류, 압력])")
    print(f"Output Shape (y): {y_final.shape} -> (샘플수, 시간스텝, [분사율])")

    # 저장
    np.savez_compressed(OUTPUT_FILENAME, X=X_final, y=y_final)
    print(f"데이터셋 저장 완료: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()