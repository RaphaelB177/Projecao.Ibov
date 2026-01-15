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
    start_str = (hoje - timedelta(days=365*12)).strftime('%Y-%m-%d')
    
    # 1. Ibov com tratamento robusto de colunas (Resolvendo o KeyError anterior)
    ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
    
    if ibov_raw.empty:
        st.error("Erro: Yahoo Finance não retornou dados para ^BVSP.")
        st.stop()

    if isinstance(ibov_raw.columns, pd.MultiIndex):
        nivel_0 = ibov_raw.columns.get_level_values(0)
        col = 'Adj Close' if 'Adj Close' in nivel_0 else 'Close' if 'Close' in nivel_0 else ibov_raw.columns[0][0]
        ibov = ibov_raw[col].iloc[:, 0]
    else:
        col = 'Adj Close' if 'Adj Close' in ibov_raw.columns else 'Close' if 'Close' in ibov_raw.columns else ibov_raw.columns[0]
        ibov = ibov_raw[col]

    # Cria o DataFrame base com DatetimeIndex
    df = ibov.resample('ME').last().to_frame('ibov')
    df.index = pd.to_datetime(df.index)

    # 2. Função SGS com tratamento para evitar o TypeError no resample
    def get_sgs_safe(codigo, nome):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200: return pd.DataFrame()
            
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            d = d.rename(columns={'valor': nome}).set_index('data')
            
            # Só faz resample se o índice for datetime e não estiver vazio
            if not d.empty and isinstance(d.index, pd.DatetimeIndex):
                return d.resample('ME').last()
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    # Join das variáveis macro (Protegido contra DataFrames vazios)
    for cod, nome in [(1, 'dolar'), (433, 'inflacao'), (4390, 'juros_brasil'), (438, 'pib')]:
        macro_data = get_sgs_safe(cod, nome)
        if not macro_data.empty:
            df = df.join(macro_data, how='left')
    
    # Preenchimento e Alvos
    df = df.ffill().dropna()
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    
    return df

df_full = load_data_v3()

# -------------------
# Interface e Modelagem (Lógica Original de Backtest)
# -------------------
st.title("📈 Projeção Multi-Horizonte Ibovespa")

st.sidebar.header("Configuração Estatística")
window_type = st.sidebar.selectbox("Janela de Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Meses da Janela Móvel", 12, 60, 36)

features = ["juros_brasil", "dolar", "pib", "inflacao"]
features_disponiveis = [f for f in features if f in df_full.columns]

# Inputs do Usuário
st.sidebar.header("Cenário de Projeção")
u_inputs = []
for f in features_disponiveis:
    val = st.sidebar.number_input(f"Expectativa para {f}", value=float(df_full[f].iloc[-1]))
    u_inputs.append(val)

# Loop de Horizontes
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
alphas = {"1 Mês": 0.5, "6 Meses": 0.5, "12 Meses": 1.0}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, col_target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[col_target])
        
        if len(df_h) < 48:
            st.warning(f"Dados insuficientes para calcular o horizonte {label}.")
            continue

        X = df_h[features_disponiveis]
        y = df_h[col_target]
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Backtest (Lógica Original)
        start_idx = 48
        preds_bt, actuals_bt = [], []
        
        for j in range(start_idx, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            
            mdl = Ridge(alpha=alphas[label]).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        res_bt = pd.DataFrame({"Real": actuals_bt, "Prev": preds_bt}, index=y.index[start_idx:])
        
        # UI
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(res_bt["Real"].cumsum(), label="Real (Acumulado)", color="black")
            ax.plot(res_bt["Prev"].cumsum(), label="Modelo (Acumulado)", color="blue", ls="--")
            ax.set_title(f"Backtest {label}")
            ax.legend()
            st.pyplot(fig)
        
        with c2:
            rmse = np.sqrt(mean_squared_error(res_bt["Real"], res_bt["Prev"]))
            st.metric("RMSE", f"{rmse:.4f}")
            
            mdl_final = Ridge(alpha=alphas[label]).fit(X_s, y)
            pred_u = mdl_final.predict(scaler.transform([u_inputs]))[0]
            
            st.write(f"**Projeção {label}:**")
            st.write(f"Retorno: {pred_u:.2%}")
            st.write(f"Alvo: {df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")
            
            std_resid = (res_bt["Real"] - res_bt["Prev"]).std()
            st.caption(f"Margem IC 95%: ±{1.96*std_resid:.2%}")
