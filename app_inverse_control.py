import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

# ... (기존 모델 클래스 및 로드 함수는 동일하므로 생략, 아래에 포함됨) ...

# ---------------------------------------------------------
# [설정 및 모델 로드] (기존과 동일)
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Injector Controller")
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; }
    .stSuccess { background-color: #064e3b; color: #6ee7b7; }
</style>
""", unsafe_allow_html=True)

class InjectorLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size1=64, hidden_size2=32, output_size=1):
        super(InjectorLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out)
        return out

@st.cache_resource
def load_resources():
    device = torch.device('cpu')
    try:
        lstm_model = InjectorLSTM().to(device)
        lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
        lstm_model.eval()
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        reg_model = joblib.load('final_injector_model.pkl')
        poly = joblib.load('poly_feature_transformer.pkl')
        return lstm_model, scaler_X, scaler_y, reg_model, poly, True
    except:
        return None, None, None, None, None, False

lstm_model, scaler_X, scaler_y, reg_model, poly, loaded = load_resources()

# ---------------------------------------------------------
# [핵심 함수] AI 기반 시뮬레이터 (Forward Model)
# ---------------------------------------------------------
def run_simulation(pressure, duration_us):
    # 1. 전류 파형 생성 (Peak -> Direct Drop -> Hold -> Hysteresis)
    # [수정됨] 사용자 피드백 반영: Peak -> Direct Drop -> Hold -> Hysteresis
    time = np.linspace(-0.5, 6.0, 1300)
    current = np.zeros_like(time)
    
    t_start = 0.0
    t_end = duration_us / 1000.0
    
    # --- Profile Parameters ---
    peak_amp = 11.5        # Peak Current (A)
    high_hold_amp = 6.0    # 1차 Hold Current (Flat)
    low_hold_amp = 2.5     # 2차 Hysteresis Current (Lower)
    
    rise_time = 0.3       # 0 -> Peak 도달 시간 (Fast)
    drop_time = 0.05       # Peak -> High Hold 하강 시간 (Very Fast)
    high_hold_duration = 0.2 # High Hold 유지 시간 (ms)
    transition_duration = 0.2 # High Hold → Low Hold 과도기 시간 (ms)
    # Hysteresis (Ripple)
    ripple_freq = 20.0     # 20kHz Ripple
    ripple_amp = 0.8       # 진동폭
    
    # Calculation
    t_peak = t_start + rise_time
    t_hold_start = t_peak + drop_time
    t_hysteresis_start = t_hold_start + high_hold_duration
    
    for i, t in enumerate(time):
        if t < t_start:
            current[i] = 0
            
        elif t >= t_start and t < t_end:
            # 1. Rising Edge (To Peak)
            if t < t_peak:
                current[i] = peak_amp * ((t - t_start) / rise_time)
                
            # 2. Immediate Drop (No Peak Hold) -> To High Hold
            elif t < t_hold_start:
                ratio = (t - t_peak) / drop_time
                current[i] = peak_amp - (peak_amp - high_hold_amp) * ratio
                
            # 3. High Hold Phase (Flat Current)
            elif t < t_hysteresis_start:
                current[i] = high_hold_amp

            # 3.5. High Hold → Low Hold 과도기 (Smooth Transition)
            elif t < t_hysteresis_start + transition_duration:  # transition_duration 시간동안 과도기 시간
                # 과도기 동안 선형 보간 (high_hold_amp → low_hold_amp)
                # ※ 과도기에는 진동(ripple) 없음
                ratio = (t - t_hysteresis_start) / transition_duration
                base_current = high_hold_amp - (high_hold_amp - low_hold_amp) * ratio
                current[i] = base_current

            # 4. Hysteresis Phase (Lower Current + Oscillation)
            else:
                # 기본 베이스 전류 (Low Hold)
                base_current = low_hold_amp
                # Hysteresis Ripple (톱니파/사인파)
                ripple = ripple_amp * np.sin(2 * np.pi * ripple_freq * (t - t_hysteresis_start))
                current[i] = base_current + ripple
                
        # 5. Shutdown
        elif t >= t_end:
            decay_period = t - t_end
            if i > 0 and current[i-1] > 0.1:
                 # 인덕턴스로 인한 소멸 (Exponential Decay)
                 current[i] = current[i-1] * 0.85
            else:
                current[i] = 0
                
    current_wave = np.maximum(current, 0)
    
    # 2. AI 추론 (LSTM)
    p_wave = np.full_like(current_wave, pressure)
    inp = scaler_X.transform(np.stack([current_wave, p_wave], axis=1))
    with torch.no_grad():
        out = lstm_model(torch.tensor(inp, dtype=torch.float32).unsqueeze(0))
    roi_lstm = np.maximum(scaler_y.inverse_transform(out.numpy()[0]).flatten(), 0)
    
    # 3. Hybrid Correction
    X_poly = poly.transform(pd.DataFrame({'Pressure_bar': [pressure], 'ET_us': [duration_us]}))
    target_mass = max(0, reg_model.predict(X_poly)[0])
    
    lstm_sum = np.sum(roi_lstm)
    corr_ratio = (target_mass / (lstm_sum * 0.05)) if lstm_sum > 0 else 1.0
    roi_final = roi_lstm * corr_ratio
    mass_final = np.sum(roi_final) * 0.05
    
    return time, current_wave, roi_final, mass_final

# ---------------------------------------------------------
# [핵심 로직] 역방향 솔버 (Inverse Solver)
# ---------------------------------------------------------
def solve_for_duration(target_mass, pressure):
    # 이진 탐색 (Binary Search) 범위 설정
    low, high = 250, 6000 # us
    best_duration = 0
    best_error = float('inf')
    iterations = 0
    
    # 10번만 반복해도 오차 0.1% 이내로 수렴함
    for _ in range(15):
        mid = (low + high) / 2
        _, _, _, mass_pred = run_simulation(pressure, mid)
        
        error = mass_pred - target_mass
        
        if abs(error) < 0.01: # 0.01mg 오차 이내면 종료
            best_duration = mid
            break
            
        if error < 0: # 목표보다 적게 쏨 -> 시간 늘려야 함
            low = mid
        else: # 목표보다 많이 쏨 -> 시간 줄여야 함
            high = mid
            
        best_duration = mid
        iterations += 1
        
    return best_duration, iterations

# ---------------------------------------------------------
# 4. UI 구성
# ---------------------------------------------------------
st.title("🤖 AI Inverse Controller")
st.markdown("Desired Mass (Output) → **AI Solver** → Required Current (Input)")

if not loaded:
    st.error("Model files not found.")
    st.stop()

# 탭 구성
tab1, tab2 = st.tabs(["🎮 Forward Control (Manual)", "🎯 Inverse Control (Auto)"])

# [Tab 1] 기존 수동 제어
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        p_manual = st.slider("Rail Pressure (bar)", 100, 300, 200, 10, key="p1")
    with col_b:
        d_manual = st.slider("Energizing Time (us)", 250, 5000, 1500, 50, key="d1")
        
    t, i, roi, mass = run_simulation(p_manual, d_manual)
    
    st.metric("Predicted Mass", f"{mass:.2f} mg")
    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=t, y=i, name="Current", line=dict(color='#fbbf24')), secondary_y=False)
    fig1.add_trace(go.Scatter(x=t, y=roi, name="ROI", fill='tozeroy', line=dict(color='#3b82f6')), secondary_y=True)
    fig1.update_layout(template="plotly_dark", height=400, margin=dict(t=30,b=20))
    st.plotly_chart(fig1, use_container_width=True)


# [Tab 2] 역방향 자동 제어
with tab2:
    st.info("💡 목표 분사량을 입력하면, AI가 필요한 **통전 시간(Duration)**과 **전류 파형**을 찾아줍니다.")
    
    col_c, col_d = st.columns(2)
    with col_c:
        target_p = st.slider("Rail Pressure (bar)", 100, 300, 200, 10, key="p2")
    with col_d:
        # 사용자가 원하는 목표값 입력
        target_m = st.number_input("Target Injection Mass (mg)", min_value=0.5, max_value=100.0, value=15.0, step=0.5)

    if st.button("🚀 Calculate Control Parameters"):
        with st.spinner("AI is optimizing control parameters..."):
            # 솔버 실행
            opt_duration, iters = solve_for_duration(target_m, target_p)
            
            # 결과 시뮬레이션
            t_opt, i_opt, roi_opt, mass_opt = run_simulation(target_p, opt_duration)
            
            # 결과 표시
            st.success(f" Optimization Complete! (Converged in {iters} iterations)")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Target Mass", f"{target_m:.2f} mg")
            c2.metric("Achieved Mass", f"{mass_opt:.2f} mg", f"Error: {mass_opt-target_m:.3f} mg")
            c3.metric("Required ET (Duration)", f"{opt_duration:.1f} μs", "Control Input")
            
            # 그래프
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            # 전류 (제어 입력)
            fig2.add_trace(go.Scatter(
                x=t_opt, y=i_opt, name="Required Current",
                line=dict(color='#34d399', width=3) # Green for solution
            ), secondary_y=False)
            
            # 분사율 (예측 결과)
            fig2.add_trace(go.Scatter(
                x=t_opt, y=roi_opt, name="Expected ROI",
                fill='tozeroy', line=dict(color='#3b82f6')
            ), secondary_y=True)
            
            fig2.update_layout(
                title=f"<b>Optimized Control Profile</b> for {target_m}mg @ {target_p}bar",
                template="plotly_dark", height=500
            )
            fig2.update_yaxes(title_text="Current (A)", secondary_y=False)
            fig2.update_yaxes(title_text="Rate (mg/ms)", secondary_y=True)
            
            st.plotly_chart(fig2, use_container_width=True)