import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib # 모델 저장용

# 1. 데이터 로드 및 전체 데이터로 재학습
df = pd.read_csv('injection_data_master_v2.csv')
df = df.dropna(subset=['Injection_Mass_mg'])

X = df[['Pressure_bar', 'ET_us']]
y = df['Injection_Mass_mg']

# 2차 다항 변환
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 최종 모델 학습 (전체 데이터 사용)
final_model = LinearRegression()
final_model.fit(X_poly, y)

print(f"최종 모델 학습 완료. (데이터 수: {len(df)}개)")
print(f"Intercept: {final_model.intercept_:.4f}")
print(f"Coefficients: {final_model.coef_}")

# 2. 모델 저장 (나중에 다른곳에서 쓰기 위해)
joblib.dump(final_model, 'final_injector_model.pkl')
joblib.dump(poly, 'poly_feature_transformer.pkl')
print("모델이 'final_injector_model.pkl'로 저장되었습니다.")

# 3. 가상 룩업 테이블 (Virtual Map) 생성
# 압력: 100 ~ 350 bar (50 bar 간격)
# 시간: 250 ~ 5000 us (50 us 간격) - 아주 촘촘하게
pressures = np.arange(100, 301, 50) 
times = np.arange(250, 5001, 50)

# 모든 조합 생성
grid_pressure, grid_time = np.meshgrid(pressures, times)
flat_pressure = grid_pressure.flatten()
flat_time = grid_time.flatten()

X_virtual = pd.DataFrame({
    'Pressure_bar': flat_pressure,
    'ET_us': flat_time
})

# 예측 수행
X_virtual_poly = poly.transform(X_virtual)
pred_mass = final_model.predict(X_virtual_poly)

# 결과 정리
X_virtual['Predicted_Mass_mg'] = pred_mass
# 마이너스 값(물리적으로 불가능)은 0으로 보정
X_virtual.loc[X_virtual['Predicted_Mass_mg'] < 0, 'Predicted_Mass_mg'] = 0

# 4. 보기 좋은 Pivot Table 형태로 변환 (행: 시간, 열: 압력)
virtual_map = X_virtual.pivot(index='ET_us', columns='Pressure_bar', values='Predicted_Mass_mg')

# CSV 저장
virtual_map.to_csv('Virtual_Injection_Map.csv')
print("\n[완료] 'Virtual_Injection_Map.csv' 파일이 생성되었습니다.")
print("이 파일은 100bar부터 350bar까지 50bar 간격의 예측 데이터를 포함합니다.")