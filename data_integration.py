import os
import pandas as pd
import re
import glob

# ==========================================
# 사용자 설정
# ==========================================
ROOT_DIR = '.' 
FILE_PATTERN = "**/*fftSmooth_shifted.lvm"
OUTPUT_FILENAME = "injection_data_master_v2.csv"

# ==========================================
# 데이터 추출 로직
# ==========================================

def extract_parameters_from_lvm(filepath):
    """
    개별 lvm 파일을 읽어서 Start 1, Start 2, Mass 값을 추출합니다.
    (띄어쓰기 유무에 상관없이 처리하도록 개선됨)
    """
    try:
        # LVM 파일 읽기 (탭 구분)
        df = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        
        if df.shape[1] < 10:
            return None

        # 9번째 컬럼(index 8)이 라벨, 10번째 컬럼(index 9)이 값
        labels = df.iloc[:, 8].astype(str).values
        values = df.iloc[:, 9].values

        extracted_data = {
            "Start 1": None,
            "Start 2": None,
            "Mass": None
        }

        for i, label in enumerate(labels):
            # 문자열 전처리: 공백 제거 및 소문자 변환 (비교를 확실하게 하기 위함)
            # 예: "Start 1" -> "start1", "Start1" -> "start1"
            clean_label = label.replace(" ", "").lower()
            
            # 값 추출 (값이 문자열일 수 있으므로 float 변환 시도)
            try:
                val = float(values[i])
            except (ValueError, TypeError):
                continue

            if "start1" in clean_label:
                extracted_data["Start 1"] = val
            elif "start2" in clean_label:
                extracted_data["Start 2"] = val
            # "kdt" 또는 "mg"가 포함되어 있으면 질량으로 인식
            elif "kdt" in clean_label or "mg" in clean_label:
                extracted_data["Mass"] = val
        
        return extracted_data

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def parse_filename_info(filepath):
    path_str = str(filepath)
    
    # Pressure 추출 (100bar, 100bar 등)
    pressure_match = re.search(r'(\d+)\s*bar', path_str, re.IGNORECASE)
    pressure = int(pressure_match.group(1)) if pressure_match else None
    
    # ET 추출 (250us, 250us 등)
    et_match = re.search(r'(\d+)\s*us', path_str, re.IGNORECASE)
    et = int(et_match.group(1)) if et_match else None

    return pressure, et

def main():
    print("데이터 추출을 시작합니다 (v2)...")
    
    search_path = os.path.join(ROOT_DIR, FILE_PATTERN)
    files = glob.glob(search_path, recursive=True)
    
    print(f"총 {len(files)}개의 파일을 찾았습니다.")
    
    data_list = []
    
    # 무시된 파일 추적
    ignored_files = {
        "no_pressure_or_et": [],  # Pressure 또는 ET 추출 실패
        "params_extraction_failed": [],  # 파라미터 추출 실패 (None 반환)
        "no_valid_params": [],  # 파라미터는 추출했지만 Mass와 Start 1이 모두 None
        "read_error": []  # 파일 읽기 에러
    }

    for filepath in files:
        pressure, et = parse_filename_info(filepath)
        
        # Pressure 또는 ET 추출 실패
        if pressure is None or et is None:
            ignored_files["no_pressure_or_et"].append({
                "file": filepath,
                "pressure": pressure,
                "et": et
            })
            continue
            
        params = extract_parameters_from_lvm(filepath)
        
        # 파라미터 추출 실패
        if params is None:
            ignored_files["params_extraction_failed"].append({
                "file": filepath,
                "pressure": pressure,
                "et": et
            })
            continue
        
        # 파라미터는 추출했지만 Mass와 Start 1이 모두 None
        if params["Mass"] is None and params["Start 1"] is None:
            ignored_files["no_valid_params"].append({
                "file": filepath,
                "pressure": pressure,
                "et": et,
                "params": params
            })
            continue
        
        # 성공적으로 처리된 데이터
        row = {
            "Pressure_bar": pressure,
            "ET_us": et,
            "Start_Delay_ms": params["Start 1"],
            "End_Delay_ms": params["Start 2"],
            "Injection_Mass_mg": params["Mass"],
            "File_Path": filepath
        }
        data_list.append(row)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"데이터 처리 결과")
    print(f"{'='*60}")
    print(f"✓ 성공적으로 처리된 데이터: {len(data_list)}개")
    print(f"✗ 무시된 파일: {sum(len(v) for v in ignored_files.values())}개\n")
    
    # 무시된 파일 상세 정보 출력
    if any(ignored_files.values()):
        print(f"{'='*60}")
        print(f"무시된 데이터 포인트 상세 정보")
        print(f"{'='*60}\n")
        
        # 1. Pressure/ET 추출 실패
        if ignored_files["no_pressure_or_et"]:
            print(f"[1] Pressure 또는 ET 추출 실패: {len(ignored_files['no_pressure_or_et'])}개")
            for item in ignored_files["no_pressure_or_et"]:
                print(f"  - {item['file']}")
                print(f"    → Pressure: {item['pressure']}, ET: {item['et']}")
            print()
        
        # 2. 파라미터 추출 실패
        if ignored_files["params_extraction_failed"]:
            print(f"[2] 파라미터 추출 실패 (파일 읽기 오류 또는 컬럼 부족): {len(ignored_files['params_extraction_failed'])}개")
            for item in ignored_files["params_extraction_failed"]:
                print(f"  - {item['file']}")
                print(f"    → Pressure: {item['pressure']} bar, ET: {item['et']} us")
            print()
        
        # 3. 유효한 파라미터 없음
        if ignored_files["no_valid_params"]:
            print(f"[3] 유효한 파라미터 없음 (Mass와 Start 1이 모두 None): {len(ignored_files['no_valid_params'])}개")
            for item in ignored_files["no_valid_params"]:
                print(f"  - {item['file']}")
                print(f"    → Pressure: {item['pressure']} bar, ET: {item['et']} us")
                print(f"    → 추출된 파라미터: Start 1={item['params']['Start 1']}, Start 2={item['params']['Start 2']}, Mass={item['params']['Mass']}")
            print()
        
        # 요약 통계
        print(f"{'='*60}")
        print(f"무시된 파일 요약")
        print(f"{'='*60}")
        print(f"  - Pressure/ET 추출 실패: {len(ignored_files['no_pressure_or_et'])}개")
        print(f"  - 파라미터 추출 실패: {len(ignored_files['params_extraction_failed'])}개")
        print(f"  - 유효한 파라미터 없음: {len(ignored_files['no_valid_params'])}개")
        print(f"  - 총 무시된 파일: {sum(len(v) for v in ignored_files.values())}개")
    
    if data_list:
        result_df = pd.DataFrame(data_list)
        result_df = result_df.sort_values(by=["Pressure_bar", "ET_us"])
        
        # Hydraulic Delay 계산 (Start 1 자체가 Delay라면 그대로 사용, 필요시 Start 2 - Start 1 = Duration 계산 가능)
        # 여기서는 원본 데이터를 그대로 저장합니다.
        
        result_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n{'='*60}")
        print(f"[성공] 데이터 추출 완료!")
        print(f"{'='*60}")
        print(f"결과 파일: {os.path.abspath(OUTPUT_FILENAME)}")
        print(f"\n--- 데이터 미리보기 ---")
        print(result_df.head())
    else:
        print("\n[경고] 데이터를 추출하지 못했습니다.")

if __name__ == "__main__":
    main()