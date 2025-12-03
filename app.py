import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

# ---------------------------------------------------------
# 1. 설정 및 스타일
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Injector AI Digital Twin")
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; }
    .stFileUploader { background-color: #1e293b; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 모델 정의 및 로드
# ---------------------------------------------------------
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

class CurrentWaveformPredictor(nn.Module):
    """ET로부터 전류 파형을 예측하는 DNN 모델"""
    def __init__(self, input_size=1, hidden_size1=256, hidden_size2=512, output_seq_len=1300, num_layers=3):
        super(CurrentWaveformPredictor, self).__init__()
        self.output_seq_len = output_seq_len
        
        # Input layer: ET (1) -> hidden_size1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        
        if num_layers == 3:
            # Hidden layer 1: hidden_size1 -> hidden_size2
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            # Hidden layer 2: hidden_size2 -> hidden_size2
            self.fc3 = nn.Linear(hidden_size2, hidden_size2)
            # Output layer: hidden_size2 -> 1300
            self.fc_out = nn.Linear(hidden_size2, output_seq_len)
        else:
            # 2 layers: hidden_size1 -> output_seq_len
            self.fc2 = None
            self.fc3 = None
            self.fc_out = nn.Linear(hidden_size1, output_seq_len)
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()  # GELU activation function
        
    def forward(self, x):
        # x: (batch_size, 1) -> [ET]
        
        # Input layer with activation
        x = self.fc1(x)
        x = self.activation(x)  # Activation: GELU
        x = self.dropout(x)
        
        if self.num_layers == 3:
            # Hidden layer 1 with activation
            x = self.fc2(x)
            x = self.activation(x)  # Activation: GELU
            x = self.dropout(x)
            
            # Hidden layer 2 with activation
            x = self.fc3(x)
            x = self.activation(x)  # Activation: GELU
            x = self.dropout(x)
        
        # Output layer: directly output 1300 points (no activation for regression)
        out = self.fc_out(x)
        return out

@st.cache_resource
def load_resources():
    device = torch.device('cpu')
    try:
        # ROI 예측 모델 로드
        lstm_model = InjectorLSTM().to(device)
        lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
        lstm_model.eval()
        
        # 전류 파형 예측 모델 로드 (선택적)
        current_model = None
        current_scaler_X = None
        current_scaler_y = None
        try:
            current_model = CurrentWaveformPredictor(input_size=1, hidden_size1=256, hidden_size2=512, output_seq_len=1300, num_layers=3).to(device)
            current_model.load_state_dict(torch.load('current_waveform_model.pth', map_location=device))
            current_model.eval()
            current_scaler_X = joblib.load('current_scaler_X.pkl')
            current_scaler_y = joblib.load('current_scaler_y.pkl')
        except:
            pass  # 전류 모델이 없어도 앱은 동작해야 함
        
        # 스케일러 & 회귀모델 로드
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        reg_model = joblib.load('final_injector_model.pkl')
        poly = joblib.load('poly_feature_transformer.pkl')
        
        return lstm_model, scaler_X, scaler_y, reg_model, poly, current_model, current_scaler_X, current_scaler_y, True
    except Exception as e:
        return None, None, None, None, None, None, None, None, False

lstm_model, scaler_X, scaler_y, reg_model, poly, current_model, current_scaler_X, current_scaler_y, loaded = load_resources()

# ---------------------------------------------------------
# 3. 입력 데이터 처리 로직 (핵심)
# ---------------------------------------------------------
# AI 모델을 사용한 전류 파형 예측
def predict_current_waveform(duration_us, total_points=1300):
    """AI 모델을 사용하여 ET로부터 전류 파형 예측"""
    if current_model is None or current_scaler_X is None or current_scaler_y is None:
        return None, None
    
    try:
        device = torch.device('cpu')
        
        # 입력 데이터 준비 (ET만 사용)
        input_meta = np.array([[duration_us]])  # (1, 1)
        input_scaled = current_scaler_X.transform(input_meta)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        
        # 예측
        with torch.no_grad():
            output_tensor = current_model(input_tensor)
        
        # Inverse transform
        output_scaled = output_tensor.cpu().numpy()  # (1, 1300)
        output_unscaled = current_scaler_y.inverse_transform(output_scaled.reshape(-1, 1)).reshape(output_scaled.shape)
        current_waveform = np.maximum(output_unscaled[0], 0)  # 음수 제거
        
        time_axis = np.linspace(-0.5, 6.0, total_points)
        return time_axis, current_waveform
    except Exception as e:
        st.error(f"전류 파형 예측 오류: {e}")
        return None, None

# [수정됨] 사용자 피드백 반영: Peak -> Direct Drop -> Hold -> Hysteresis
def generate_realistic_waveform(duration_us, total_points=1300):
    time = np.linspace(-0.5, 6.0, total_points)
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
                
    return time, np.maximum(current, 0)

def process_uploaded_file(uploaded_file, target_points=1300):
    try:
        try:
            df = pd.read_csv(uploaded_file, sep='\t', header=None, engine='python')
        except:
            df = pd.read_csv(uploaded_file, sep=',', header=None, engine='python')
        
        if df.shape[0] < 100: return None, None
        
        # Shifted 파일 포맷 가정
        raw_time = df.iloc[:, 0].values
        raw_current = df.iloc[:, 3].values 
        
        model_time = np.linspace(-0.5, 6.0, target_points)
        f = interp1d(raw_time, raw_current, kind='linear', bounds_error=False, fill_value=0)
        resampled_current = f(model_time)
        return model_time, resampled_current
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None


# ---------------------------------------------------------
# 4. UI 구성
# ---------------------------------------------------------
st.title("⚡ True Digital Twin: Real Input Integration")
st.markdown("Physics-Informed AI with **Real/Realistic Current Inputs**")

if not loaded:
    st.error("모델 파일이 없습니다. 학습 코드를 먼저 실행해주세요.")
    st.stop()

# 사이드바
st.sidebar.header("🎛️ Input Source")
input_mode = st.sidebar.radio("Select Input Mode", [
    "AI Prediction (ET → Current)",
    "Simulation (Hysteresis)",
    "Upload Real File (.lvm)"
])

pressure = st.sidebar.slider("Rail Pressure (bar)", 100, 300, 200, 10)

current_wave = None
time_axis = None
duration_val = 0

# [로직 분기 1] AI 예측 모드
if input_mode == "AI Prediction (ET → Current)":
    duration_val = st.sidebar.slider("Energizing Time (us)", 250, 5000, 2500, 50)
    time_axis, current_wave = predict_current_waveform(duration_val)
    if current_wave is None:
        st.sidebar.warning("⚠️ 전류 파형 예측 모델이 없습니다. 수동 시뮬레이션 모드를 사용하거나 모델을 학습해주세요.")
        # Fallback to manual simulation
        time_axis, current_wave = generate_realistic_waveform(duration_val)
        st.sidebar.caption("⚠️ 수동 시뮬레이션 모드로 전환됨")
    else:
        st.sidebar.success(f"✅ AI 예측: P={pressure}bar, ET={duration_val}us")

# [로직 분기 2] 시뮬레이션 모드
elif input_mode == "Simulation (Hysteresis)":
    duration_val = st.sidebar.slider("Energizing Time (us)", 250, 5000, 2500, 50)
    time_axis, current_wave = generate_realistic_waveform(duration_val)
    st.sidebar.caption("✅ Peak -> Fast Drop -> Hysteresis 패턴 적용됨")

# [로직 분기 3] 파일 업로드 모드
else:
    uploaded_file = st.sidebar.file_uploader("Upload Current Data", type=['lvm', 'txt', 'csv'])
    if uploaded_file is not None:
        time_axis, current_wave = process_uploaded_file(uploaded_file)
        if current_wave is not None:
            # 파일에서 대략적인 ET 추정 (0A 이상인 구간)
            mask = current_wave > 1.0 
            if np.any(mask):
                duration_val = (time_axis[mask][-1] - time_axis[mask][0]) * 1000
            else:
                duration_val = 0
            st.sidebar.success(f"File Loaded! Est. ET: {duration_val:.0f} us")
    
    if current_wave is None:
        st.info("👈 Please upload a '.lvm' file containing current data (Shifted format).")
        st.stop()

# ---------------------------------------------------------
# 5. AI 추론 및 하이브리드 보정
# ---------------------------------------------------------
# A. LSTM 추론
pressure_wave = np.full_like(current_wave, pressure)
input_raw = np.stack([current_wave, pressure_wave], axis=1)
input_scaled = scaler_X.transform(input_raw)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    output_tensor = lstm_model(input_tensor)

output_raw = scaler_y.inverse_transform(output_tensor.numpy()[0]).flatten()
output_lstm = np.maximum(output_raw, 0)

# B. 회귀 모델을 이용한 Mass 보정
# (파일 업로드 시에도 Pressure 정보는 슬라이더 값을 사용한다고 가정 - 또는 파일명에서 파싱 가능)
X_reg = pd.DataFrame({'Pressure_bar': [pressure], 'ET_us': [duration_val]})
X_poly = poly.transform(X_reg)
target_mass = reg_model.predict(X_poly)[0]
target_mass = max(0, target_mass)

lstm_integral = np.sum(output_lstm)
# 보정 로직 (비율 기반)
if lstm_integral > 0:
    # 학습 데이터 기준 스케일 상수 (임의값) - 실제론 dt 고려해야 함
    # 여기서는 LSTM값 자체를 신뢰하되, Mass 비율만 맞춤
    lstm_mass_est = lstm_integral * 0.05 
    correction_ratio = target_mass / lstm_mass_est if lstm_mass_est > 0 else 1.0
else:
    correction_ratio = 1.0

# 하이브리드 결과
output_hybrid = output_lstm * correction_ratio
final_mass = np.sum(output_hybrid) * 0.05

# ---------------------------------------------------------
# 6. 시각화
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
source_label = "AI Prediction" if input_mode == "AI Prediction (ET → Current)" else ("Real File" if input_mode.startswith("Upload") else "Simulation")
col1.metric("Input Source", source_label)
col2.metric("Total Mass (Hybrid)", f"{final_mass:.2f} mg", f"Target: {target_mass:.2f} mg")
col3.metric("Current Peak", f"{np.max(current_wave):.1f} A", f"Hold: ~6.0 A")

fig = make_subplots(specs=[[{"secondary_y": True}]])

# 1. 전류 파형 (입력)
fig.add_trace(go.Scatter(
    x=time_axis, y=current_wave, 
    name="Input Current (Real/Sim)",
    line=dict(color='#fbbf24', width=2) # Amber color
), secondary_y=False)

# 2. 분사율 (출력)
fig.add_trace(go.Scatter(
    x=time_axis, y=output_hybrid, 
    name="AI Predicted ROI",
    fill='tozeroy',
    line=dict(color='#3b82f6', width=3) # Blue
), secondary_y=True)

fig.update_layout(
    title="<b>Current (Input)</b> vs <b>ROI (Output)</b>",
    template="plotly_dark",
    hovermode="x unified",
    height=500,
    legend=dict(orientation="h", y=1.1)
)

fig.update_yaxes(title_text="Current (A)", secondary_y=False)
fig.update_yaxes(title_text="Injection Rate (mg/ms)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# 설명
if input_mode == "AI Prediction (ET → Current)":
    st.success("🤖 **AI Prediction Mode:** 실제 측정 데이터로 학습된 모델을 사용하여 전류 파형을 예측합니다.")
elif input_mode.startswith("Simulation"):
    st.info("💡 **Hysteresis Simulation:** Peak(12A) → Drop → Hold(6A) with PWM Ripple applied.")
else:
    st.success("📂 **Real Data Mode:** Processing actual current waveform from the uploaded file.")