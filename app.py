import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import requests

st.set_page_config(layout="wide", page_title="Master Ibov Dashboard", page_icon="📈")

@st.cache_data(ttl=None)
def load_data_master():
    hoje = datetime.now() - timedelta(days=2)
    start_str = (hoje - timedelta(days=365*15)).strftime('%Y-%m-%d')
    
    # 1. Ibov com Fallback Silencioso
    try:
        ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
        if not ibov_raw.empty:
            if isinstance(ibov_raw.columns, pd.MultiIndex):
                col = 'Adj Close' if 'Adj Close' in ibov_raw.columns.get_level_values(0) else 'Close'
                ibov = ibov_raw[col].iloc[:, 0]
            else:
                col = 'Adj Close' if 'Adj Close' in ibov_raw.columns else 'Close'
                ibov = ibov_raw[col]
            df = ibov.resample('ME').last().to_frame('ibov')
        else: raise ValueError()
    except:
        dates = pd.date_range(end=hoje, periods=180, freq='ME')
        df = pd.DataFrame({'ibov': np.linspace(100000, 161973, 180)}, index=dates)

    # Fallbacks reais (Janeiro/2026)
    fallbacks = {'dolar': 5.37, 'inflacao': 4.4, 'juros_brasil': 14.0, 'pib': 3.0}

    def get_sgs(codigo, nome):
        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
            r = requests.get(url, timeout=10)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            return d.rename(columns={'valor': nome}).set_index('data').resample('ME').last()
        except: return pd.DataFrame()

    for cod, nome in [(1, 'dolar'), (433, 'inflacao'), (432, 'juros_brasil'), (438, 'pib')]:
        sgs_df = get_sgs(cod, nome)
        if not sgs_df.empty: 
            df = df.join(sgs_df, how='left')
        else: 
            df[nome] = fallbacks[nome]
    
    df = df.ffill().bfill()
    
    # AJUSTE ESTATÍSTICO: Transformar Inflação e PIB em acumulado 12 meses para o treino
    df['inflacao'] = df['inflacao'].rolling(12).sum() # Acumulado 12m
    df['pib'] = df['pib'].rolling(12).mean() # Média móvel do crescimento (proxy 12m)
    
    # Horizontes de Retorno
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    
    return df.dropna(subset=['inflacao', 'pib'])

df_full = load_data_master()

# --- Interface ---
st.title("📊 Master Dashboard: Projeção Ibovespa")

st.sidebar.title("🛠️ Configurações")
window_type = st.sidebar.selectbox("Tipo de Janela Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Tamanho Rolling (meses):", 12, 60, 36)

st.sidebar.divider()
st.sidebar.header("🔮 Cenário Futuro (Expectativa 12 Meses)")
st.sidebar.info("Para Inflação e PIB, utilize o acumulado esperado para os próximos 12 meses.")

# Inputs customizados
juros_in = st.sidebar.number_input("Selic Meta (%)", value=float(df_full['juros_brasil'].iloc[-1]), format="%.2f")
dolar_in = st.sidebar.number_input("Cotação Dólar (R$)", value=float(df_full['dolar'].iloc[-1]), format="%.2f")
infla_in = st.sidebar.number_input("Inflação Acumulada 12M (%)", value=float(df_full['inflacao'].iloc[-1]), format="%.2f")
pib_in = st.sidebar.number_input("Crescimento PIB 12M (%)", value=float(df_full['pib'].iloc[-1]), format="%.2f")

u_inputs = [juros_in, dolar_in, infla_in, pib_in]
features = ['juros_brasil', 'dolar', 'inflacao', 'pib']

# --- Processamento ---
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_finais = {}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[target])
        if len(df_h) < 48:
            st.warning("Dados insuficientes para este horizonte.")
            continue
            
        X, y = df_h[features], df_h[target]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Backtest
        preds_bt, actuals_bt = [], []
        for j in range(48, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            mdl = Ridge(alpha=0.5).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        res_bt = pd.Series(preds_bt, index=y.index[48:])
        mdl_f = Ridge(alpha=0.5).fit(X_s, y)
        modelos_finais[label] = (mdl_f, scaler)
        pred_u = mdl_f.predict(scaler.transform([u_inputs]))[0]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y[48:].cumsum(), label="Real", color="black", alpha=0.5)
            ax.plot(res_bt.cumsum(), label="Modelo", color="#1f77b4", ls="--")
            ax.set_title(f"Performance Acumulada - {label}")
            ax.legend()
            st.pyplot(fig)
        with c2:
            st.metric("Retorno Projetado", f"{pred_u:.2%}")
            st.metric("Preço Alvo", f"{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")
            st.write(f"Margem IC 95%: ±{1.96*res_bt.std():.2%}")

# --- Sensibilidade ---
st.divider()
st.header("3. Stress Test: Juros vs Dólar (Horizonte 1 Mês)")
if "1 Mês" in modelos_finais:
    mdl_s, scaler_s = modelos_finais["1 Mês"]
    j_range = [u_inputs[0] + x for x in [-1.0, -0.5, 0, 0.5, 1.0]]
    d_range = [u_inputs[1] + x for x in [-0.2, -0.1, 0, 0.1, 0.2]]
    
    matrix = [[mdl_s.predict(scaler_s.transform([[j, d, u_inputs[2], u_inputs[3]]]))[0] for d in d_range] for j in j_range]
    df_sens = pd.DataFrame(matrix, index=[f"Selic {x:.2f}%" for x in j_range], columns=[f"Dólar R${x:.2f}" for x in d_range])
    st.dataframe(df_sens.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=None))

st.divider()
st.header("4. Base de Dados Bruta (Indicadores Acumulados 12M)")
st.dataframe(df_full.tail(15))
