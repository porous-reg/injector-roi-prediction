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
# [설정 및 모델 로드]
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Injector Controller (HRM)")
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; }
    .stSuccess { background-color: #064e3b; color: #6ee7b7; }
</style>
""", unsafe_allow_html=True)

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
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        reg_model = joblib.load('final_injector_model.pkl')
        poly = joblib.load('poly_feature_transformer.pkl')
        return hrm_model, scaler_X, scaler_y, reg_model, poly, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, None, None, False

hrm_model, scaler_X, scaler_y, reg_model, poly, loaded = load_resources()

# ---------------------------------------------------------
# [핵심 함수] AI 기반 시뮬레이터 (Forward Model)
# ---------------------------------------------------------
def run_simulation(pressure, duration_us):
    time = np.linspace(-0.5, 6.0, 1300)
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
                
    current_wave = np.maximum(current, 0)
    
    # AI 추론 (HRM)
    p_wave = np.full_like(current_wave, pressure)
    inp = scaler_X.transform(np.stack([current_wave, p_wave], axis=1))
    with torch.no_grad():
        out = hrm_model(torch.tensor(inp, dtype=torch.float32).unsqueeze(0))
    roi_hrm = np.maximum(scaler_y.inverse_transform(out.numpy()[0]).flatten(), 0)
    
    # Hybrid Correction
    X_poly = poly.transform(pd.DataFrame({'Pressure_bar': [pressure], 'ET_us': [duration_us]}))
    target_mass = max(0, reg_model.predict(X_poly)[0])
    
    hrm_sum = np.sum(roi_hrm)
    corr_ratio = (target_mass / (hrm_sum * 0.05)) if hrm_sum > 0 else 1.0
    roi_final = roi_hrm * corr_ratio
    mass_final = np.sum(roi_final) * 0.05
    
    return time, current_wave, roi_final, mass_final

# ---------------------------------------------------------
# [핵심 로직] 역방향 솔버 (Inverse Solver)
# ---------------------------------------------------------
def solve_for_duration(target_mass, pressure):
    low, high = 250, 6000
    best_duration = 0
    iterations = 0
    
    for _ in range(15):
        mid = (low + high) / 2
        _, _, _, mass_pred = run_simulation(pressure, mid)
        
        error = mass_pred - target_mass
        
        if abs(error) < 0.01:
            best_duration = mid
            break
            
        if error < 0:
            low = mid
        else:
            high = mid
            
        best_duration = mid
        iterations += 1
        
    return best_duration, iterations

# ---------------------------------------------------------
# 4. UI 구성
# ---------------------------------------------------------
st.title("🤖 AI Inverse Controller (HRM)")
st.markdown("Desired Mass (Output) → **HRM AI Solver** → Required Current (Input)")

if not loaded:
    st.error("Model files not found. Please train the HRM model first.")
    st.stop()

tab1, tab2 = st.tabs(["🎮 Forward Control (Manual)", "🎯 Inverse Control (Auto)"])

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

with tab2:
    st.info("💡 목표 분사량을 입력하면, HRM AI가 필요한 **통전 시간(Duration)**과 **전류 파형**을 찾아줍니다.")
    
    col_c, col_d = st.columns(2)
    with col_c:
        target_p = st.slider("Rail Pressure (bar)", 100, 300, 200, 10, key="p2")
    with col_d:
        target_m = st.number_input("Target Injection Mass (mg)", min_value=0.5, max_value=100.0, value=15.0, step=0.5)

    if st.button("🚀 Calculate Control Parameters"):
        with st.spinner("HRM AI is optimizing control parameters..."):
            opt_duration, iters = solve_for_duration(target_m, target_p)
            
            t_opt, i_opt, roi_opt, mass_opt = run_simulation(target_p, opt_duration)
            
            st.success(f"Optimization Complete! (Converged in {iters} iterations)")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Target Mass", f"{target_m:.2f} mg")
            c2.metric("Achieved Mass", f"{mass_opt:.2f} mg", f"Error: {mass_opt-target_m:.3f} mg")
            c3.metric("Required ET (Duration)", f"{opt_duration:.1f} μs", "Control Input")
            
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(
                x=t_opt, y=i_opt, name="Required Current",
                line=dict(color='#34d399', width=3)
            ), secondary_y=False)
            
            fig2.add_trace(go.Scatter(
                x=t_opt, y=roi_opt, name="Expected ROI",
                fill='tozeroy', line=dict(color='#3b82f6')
            ), secondary_y=True)
            
            fig2.update_layout(
                title=f"<b>Optimized Control Profile (HRM)</b> for {target_m}mg @ {target_p}bar",
                template="plotly_dark", height=500
            )
            fig2.update_yaxes(title_text="Current (A)", secondary_y=False)
            fig2.update_yaxes(title_text="Rate (mg/ms)", secondary_y=True)
            
            st.plotly_chart(fig2, use_container_width=True)

