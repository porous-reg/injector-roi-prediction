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
st.set_page_config(layout="wide", page_title="Injector AI Digital Twin (HRM)")
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; }
    .stFileUploader { background-color: #1e293b; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. HRM 모델 정의 및 로드
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
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
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

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

class InjectorHRM(nn.Module):
    """Hybrid Recursive Model: LSTM + Transformer 병렬 결합"""
    def __init__(self, input_size=2, lstm_hidden1=64, lstm_hidden2=32, 
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 output_size=1, dropout=0.1, max_len=1300):
        super(InjectorHRM, self).__init__()
        
        # LSTM 브랜치
        self.lstm1 = nn.LSTM(input_size, lstm_hidden1, batch_first=True)
        self.dropout_lstm1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(lstm_hidden1, lstm_hidden2, batch_first=True)
        self.dropout_lstm2 = nn.Dropout(dropout)
        
        # Transformer 브랜치
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.transformer_blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Feature Fusion
        fusion_dim = lstm_hidden2 + d_model
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Dropout(dropout)
        )
        
        # 출력 레이어
        self.output_proj = nn.Linear(fusion_dim // 2, output_size)
        
    def forward(self, x):
        # LSTM 브랜치
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout_lstm1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout_lstm2(lstm_out)
        
        # Transformer 브랜치
        trans_out = self.input_proj(x)
        trans_out = self.pos_encoder(trans_out)
        for transformer_block in self.transformer_blocks:
            trans_out = transformer_block(trans_out)
        
        # Feature Fusion
        fused = torch.cat([lstm_out, trans_out], dim=-1)
        fused = self.fusion(fused)
        
        # 최종 출력
        out = self.output_proj(fused)
        return out

@st.cache_resource
def load_resources():
    device = torch.device('cpu')
    try:
        # HRM 모델 로드
        hrm_model = InjectorHRM(
            input_size=2,
            lstm_hidden1=64,
            lstm_hidden2=32,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            output_size=1,
            dropout=0.1
        ).to(device)
        hrm_model.load_state_dict(torch.load('hrm_model.pth', map_location=device))
        hrm_model.eval()
        
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
        
        return hrm_model, scaler_X, scaler_y, reg_model, poly, current_model, current_scaler_X, current_scaler_y, True
    except Exception as e:
        st.error(f"모델 로드 오류: {e}")
        return None, None, None, None, None, None, None, None, False

hrm_model, scaler_X, scaler_y, reg_model, poly, current_model, current_scaler_X, current_scaler_y, loaded = load_resources()

# ---------------------------------------------------------
# 3. 입력 데이터 처리 로직
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

def generate_realistic_waveform(duration_us, total_points=1300):
    time = np.linspace(-0.5, 6.0, total_points)
    current = np.zeros_like(time)
    
    t_start = 0.0
    t_end = duration_us / 1000.0
    
    peak_amp = 11.5
    high_hold_amp = 6.0
    low_hold_amp = 2.5
    
    rise_time = 0.3
    drop_time = 0.05
    high_hold_duration = 0.2
    transition_duration = 0.2
    ripple_freq = 20.0
    ripple_amp = 0.8
    
    t_peak = t_start + rise_time
    t_hold_start = t_peak + drop_time
    t_hysteresis_start = t_hold_start + high_hold_duration
    
    for i, t in enumerate(time):
        if t < t_start:
            current[i] = 0
        elif t >= t_start and t < t_end:
            if t < t_peak:
                current[i] = peak_amp * ((t - t_start) / rise_time)
            elif t < t_hold_start:
                ratio = (t - t_peak) / drop_time
                current[i] = peak_amp - (peak_amp - high_hold_amp) * ratio
            elif t < t_hysteresis_start:
                current[i] = high_hold_amp
            elif t < t_hysteresis_start + transition_duration:
                ratio = (t - t_hysteresis_start) / transition_duration
                base_current = high_hold_amp - (high_hold_amp - low_hold_amp) * ratio
                current[i] = base_current
            else:
                base_current = low_hold_amp
                ripple = ripple_amp * np.sin(2 * np.pi * ripple_freq * (t - t_hysteresis_start))
                current[i] = base_current + ripple
        elif t >= t_end:
            if i > 0 and current[i-1] > 0.1:
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
st.title("⚡ True Digital Twin: Real Input Integration (HRM)")
st.markdown("Physics-Informed AI with **HRM (Hybrid Recursive Model)** - LSTM + Transformer Hybrid Architecture")

if not loaded:
    st.error("모델 파일이 없습니다. HRM 학습 코드를 먼저 실행해주세요.")
    st.stop()

st.sidebar.header("🎛️ Input Source")
input_mode = st.sidebar.radio("Select Input Mode", [
    "AI Prediction (ET → Current)",
    "Simulation (Hysteresis)",
    "Upload Real File (.lvm)"
])

pressure = st.sidebar.slider("Rail Pressure (bar)", 100, 350, 300, 10)

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
pressure_wave = np.full_like(current_wave, pressure)
input_raw = np.stack([current_wave, pressure_wave], axis=1)
input_scaled = scaler_X.transform(input_raw)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    output_tensor = hrm_model(input_tensor)

output_raw = scaler_y.inverse_transform(output_tensor.numpy()[0]).flatten()
output_hrm = np.maximum(output_raw, 0)

X_reg = pd.DataFrame({'Pressure_bar': [pressure], 'ET_us': [duration_val]})
X_poly = poly.transform(X_reg)
target_mass = reg_model.predict(X_poly)[0]
target_mass = max(0, target_mass)

hrm_integral = np.sum(output_hrm)
if hrm_integral > 0:
    hrm_mass_est = hrm_integral * 0.05 
    correction_ratio = target_mass / hrm_mass_est if hrm_mass_est > 0 else 1.0
else:
    correction_ratio = 1.0

output_hybrid = output_hrm * correction_ratio
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

fig.add_trace(go.Scatter(
    x=time_axis, y=current_wave, 
    name="Input Current (Real/Sim)",
    line=dict(color='#fbbf24', width=2)
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=time_axis, y=output_hybrid, 
    name="HRM Predicted ROI",
    fill='tozeroy',
    line=dict(color='#3b82f6', width=3)
), secondary_y=True)

fig.update_layout(
    title="<b>Current (Input)</b> vs <b>ROI (Output) - HRM Model</b>",
    template="plotly_dark",
    hovermode="x unified",
    height=500,
    legend=dict(orientation="h", y=1.1)
)

fig.update_yaxes(title_text="Current (A)", secondary_y=False)
fig.update_yaxes(title_text="Injection Rate (mg/ms)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

if input_mode == "AI Prediction (ET → Current)":
    st.success("🤖 **AI Prediction Mode:** 실제 측정 데이터로 학습된 모델을 사용하여 전류 파형을 예측합니다. Using **HRM (Hybrid LSTM+Transformer)** model.")
elif input_mode.startswith("Simulation"):
    st.info("💡 **Hysteresis Simulation:** Peak(12A) → Drop → Hold(6A) with PWM Ripple applied. Using **HRM (Hybrid LSTM+Transformer)** model.")
else:
    st.success("📂 **Real Data Mode:** Processing actual current waveform from the uploaded file. Using **HRM (Hybrid LSTM+Transformer)** model.")

