import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import requests

# -------------------
# Configurações e Coleta
# -------------------
st.set_page_config(layout="wide", page_title="Ibov Projeção Multi-Horizonte", page_icon="📈")

@st.cache_data(ttl=None)
def load_data_v3():
    hoje = datetime.now() - timedelta(days=2)
    # Janela de 12 anos para garantir dados suficientes para o alvo de 12m
    start_str = (hoje - timedelta(days=365*12)).strftime('%Y-%m-%d')
    
    # 1. Ibov com tratamento robusto de colunas
    ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
    
    if ibov_raw.empty:
        st.error("Erro: Yahoo Finance não retornou dados para ^BVSP.")
        st.stop()

    # Tratamento flexível de colunas (Adj Close ou Close)
    if isinstance(ibov_raw.columns, pd.MultiIndex):
        cols_nivel_0 = ibov_raw.columns.get_level_values(0)
        if 'Adj Close' in cols_nivel_0:
            ibov = ibov_raw['Adj Close'].iloc[:, 0]
        elif 'Close' in cols_nivel_0:
            ibov = ibov_raw['Close'].iloc[:, 0]
        else:
            ibov = ibov_raw.iloc[:, 0] # Pega a primeira coluna disponível
    else:
        if 'Adj Close' in ibov_raw.columns:
            ibov = ibov_raw['Adj Close']
        elif 'Close' in ibov_raw.columns:
            ibov = ibov_raw['Close']
        else:
            ibov = ibov_raw.iloc[:, 0]

    df = ibov.resample('ME').last().to_frame('ibov')

    # 2. Funções SGS (CSV Direto)
    def get_sgs(codigo, nome):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
        try:
            r = requests.get(url, timeout=15)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            return d.rename(columns={'valor': nome}).set_index('data')
        except: return pd.DataFrame()

    # Join das variáveis macro
    df = df.join(get_sgs(1, 'dolar').resample('ME').last(), how='left')
    df = df.join(get_sgs(433, 'inflacao').resample('ME').last(), how='left')
    df = df.join(get_sgs(4390, 'juros_brasil').resample('ME').last(), how='left')
    df = df.join(get_sgs(438, 'pib').resample('ME').last(), how='left')
    
    df = df.ffill().dropna()
    
    # Criando alvos para diferentes horizontes (Retorno Acumulado Futuro)
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    
    return df

df_full = load_data_v3()

# -------------------
# Interface e Modelagem
# -------------------
st.title("📈 Projeção Multi-Horizonte Ibovespa")

# Parâmetros de Backtest na Sidebar
st.sidebar.header("Configuração Estatística")
window_type = st.sidebar.selectbox("Janela de Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Meses da Janela Móvel", 12, 60, 36)

features = ["juros_brasil", "dolar", "pib", "inflacao"]
X_raw = df_full[features]

# Inputs do Usuário para Projeção
st.sidebar.header("Cenário de Projeção")
u_inputs = [st.sidebar.number_input(f"Expectativa para {f}", value=float(X_raw[f].iloc[-1])) for f in features]

# -------------------
# Loop de Horizontes
# -------------------
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
alphas = {"1 Mês": 0.5, "6 Meses": 0.5, "12 Meses": 1.0}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, col_target) in enumerate(horizontes.items()):
    with tabs[i]:
        # Filtra apenas linhas onde o alvo existe (remove os nulos do shift no final)
        df_h = df_full.dropna(subset=[col_target])
        
        if len(df_h) < 60:
            st.warning(f"Dados insuficientes para o horizonte {label}.")
            continue

        X = df_h[features]
        y = df_h[col_target]
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Backtest (Lógica original)
        start_idx = 48
        preds_bt = []
        actuals_bt = []
        
        for j in range(start_idx, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            
            mdl = Ridge(alpha=alphas[label]).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        res_bt = pd.DataFrame({"Real": actuals_bt, "Prev": preds_bt}, index=y.index[start_idx:])
        rmse = np.sqrt(mean_squared_error(res_bt["Real"], res_bt["Prev"]))
        
        # Dashboard do Horizonte
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(res_bt["Real"].cumsum(), label="Real (Acumulado)", color="black", lw=1.5)
            ax.plot(res_bt["Prev"].cumsum(), label="Previsto (Acumulado)", color="blue", ls="--")
            ax.set_title(f"Backtest Acumulado - Horizonte {label}")
            ax.legend()
            st.pyplot(fig)
        
        with c2:
            st.metric(f"RMSE ({label})", f"{rmse:.4f}")
            # Projeção Real-Time
            mdl_final = Ridge(alpha=alphas[label]).fit(X_s, y)
            u_scaled = scaler.transform([u_inputs])
            pred_u = mdl_final.predict(u_scaled)[0]
            
            st.subheader("Resultado da Projeção")
            st.write(f"Retorno esperado: **{pred_u:.2%}**")
            st.write(f"Preço alvo: **{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}**")
            
            std_resid = (res_bt["Real"] - res_bt["Prev"]).std()
            st.caption(f"Margem de Erro (95%): ±{1.96*std_resid:.2%}")

st.divider()
with st.expander("Ver base de dados processada"):
    st.write(df_full.tail(10))
