import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import math

# ---------------------------------------------------------
# 1. 설정 및 스타일
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Injector AI Digital Twin (TRM)")
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; }
    .stFileUploader { background-color: #1e293b; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. TRM 모델 정의 및 로드
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Positional Encoding for Time Series"""
    def __init__(self, d_model, max_len=1300, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TinyTransformerBlock(nn.Module):
    """Single Transformer Block with Self-Attention"""
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TinyTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class InjectorTRM(nn.Module):
    """Tiny Recursive Model (Transformer-based) for Injector ROI Prediction"""
    def __init__(self, input_size=2, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, output_size=1, dropout=0.1, max_len=1300):
        super(InjectorTRM, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
        
        self.d_model = d_model
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Output projection
        out = self.output_proj(x)  # (batch, seq_len, output_size)
        
        return out

@st.cache_resource
def load_resources():
    device = torch.device('cpu')
    try:
        # TRM 모델 로드
        trm_model = InjectorTRM(
            input_size=2,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            output_size=1,
            dropout=0.1
        ).to(device)
        trm_model.load_state_dict(torch.load('trm_model.pth', map_location=device))
        trm_model.eval()
        
        # 스케일러 & 회귀모델 로드
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        reg_model = joblib.load('final_injector_model.pkl')
        poly = joblib.load('poly_feature_transformer.pkl')
        
        return trm_model, scaler_X, scaler_y, reg_model, poly, True
    except Exception as e:
        st.error(f"모델 로드 오류: {e}")
        return None, None, None, None, None, False

trm_model, scaler_X, scaler_y, reg_model, poly, loaded = load_resources()

# ---------------------------------------------------------
# 3. 입력 데이터 처리 로직 (핵심)
# ---------------------------------------------------------
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
st.title("⚡ True Digital Twin: Real Input Integration (TRM)")
st.markdown("Physics-Informed AI with **TRM (Tiny Recursive Model)** - Transformer-based Architecture")

if not loaded:
    st.error("모델 파일이 없습니다. TRM 학습 코드를 먼저 실행해주세요.")
    st.stop()

# 사이드바
st.sidebar.header("🎛️ Input Source")
input_mode = st.sidebar.radio("Select Input Mode", ["Simulation (Hysteresis)", "Upload Real File (.lvm)"])

pressure = st.sidebar.slider("Rail Pressure (bar)", 100, 350, 300, 10)

current_wave = None
time_axis = None
duration_val = 0

# [로직 분기 1] 시뮬레이션 모드
if input_mode == "Simulation (Hysteresis)":
    duration_val = st.sidebar.slider("Energizing Time (us)", 250, 5000, 2500, 50)
    time_axis, current_wave = generate_realistic_waveform(duration_val)
    st.sidebar.caption("✅ Peak -> Fast Drop -> Hysteresis 패턴 적용됨")

# [로직 분기 2] 파일 업로드 모드
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
# A. TRM 추론
pressure_wave = np.full_like(current_wave, pressure)
input_raw = np.stack([current_wave, pressure_wave], axis=1)
input_scaled = scaler_X.transform(input_raw)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    output_tensor = trm_model(input_tensor)

output_raw = scaler_y.inverse_transform(output_tensor.numpy()[0]).flatten()
output_trm = np.maximum(output_raw, 0)

# B. 회귀 모델을 이용한 Mass 보정
# (파일 업로드 시에도 Pressure 정보는 슬라이더 값을 사용한다고 가정 - 또는 파일명에서 파싱 가능)
X_reg = pd.DataFrame({'Pressure_bar': [pressure], 'ET_us': [duration_val]})
X_poly = poly.transform(X_reg)
target_mass = reg_model.predict(X_poly)[0]
target_mass = max(0, target_mass)

trm_integral = np.sum(output_trm)
# 보정 로직 (비율 기반)
if trm_integral > 0:
    # 학습 데이터 기준 스케일 상수 (임의값) - 실제론 dt 고려해야 함
    # 여기서는 TRM값 자체를 신뢰하되, Mass 비율만 맞춤
    trm_mass_est = trm_integral * 0.05 
    correction_ratio = target_mass / trm_mass_est if trm_mass_est > 0 else 1.0
else:
    correction_ratio = 1.0

# 하이브리드 결과
output_hybrid = output_trm * correction_ratio
final_mass = np.sum(output_hybrid) * 0.05

# ---------------------------------------------------------
# 6. 시각화
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Input Source", "Real File" if input_mode.startswith("Upload") else "Simulation")
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
    name="TRM Predicted ROI",
    fill='tozeroy',
    line=dict(color='#3b82f6', width=3) # Blue
), secondary_y=True)

fig.update_layout(
    title="<b>Current (Input)</b> vs <b>ROI (Output) - TRM Model</b>",
    template="plotly_dark",
    hovermode="x unified",
    height=500,
    legend=dict(orientation="h", y=1.1)
)

fig.update_yaxes(title_text="Current (A)", secondary_y=False)
fig.update_yaxes(title_text="Injection Rate (mg/ms)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# 설명
if input_mode.startswith("Simulation"):
    st.info("💡 **Hysteresis Simulation:** Peak(12A) → Drop → Hold(6A) with PWM Ripple applied. Using **TRM (Transformer-based)** model.")
else:
    st.success("📂 **Real Data Mode:** Processing actual current waveform from the uploaded file. Using **TRM (Transformer-based)** model.")

