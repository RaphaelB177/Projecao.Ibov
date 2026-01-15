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

# -------------------
# Configurações Iniciais
# -------------------
st.set_page_config(layout="wide", page_title="Master Ibov Dashboard", page_icon="📈")

@st.cache_data(ttl=3600)
def load_data_master():
    hoje = datetime.now() - timedelta(days=1)
    start_str = (hoje - timedelta(days=365*15)).strftime('%Y-%m-%d')
    
    # 1. Ibov e Dólar (Yahoo Finance)
    try:
        # Ticker BRL=X é o mais estável para dólar comercial
        data_yf = yf.download(["^BVSP", "BRL=X"], start=start_str, progress=False)
        
        if isinstance(data_yf.columns, pd.MultiIndex):
            ibov = data_yf['Adj Close']['^BVSP']
            dolar = data_yf['Adj Close']['BRL=X']
        else:
            ibov = data_yf['^BVSP']
            dolar = data_yf['BRL=X']
            
        df = pd.DataFrame({'ibov': ibov, 'dolar': dolar})
        df = df.resample('ME').last()
    except:
        # Fallback caso o Yahoo falhe
        dates = pd.date_range(end=hoje, periods=180, freq='ME')
        df = pd.DataFrame({'ibov': np.nan, 'dolar': np.nan}, index=dates)

    # 2. Séries do Banco Central (SGS)
    def get_sgs(codigo, nome):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
                d['data'] = pd.to_datetime(d['data'], dayfirst=True)
                return d.rename(columns={'valor': nome}).set_index('data').resample('ME').last()
        except: pass
        return pd.DataFrame()

    # Fallbacks reais para Jan/2026 se a API falhar
    fallbacks = {'inflacao': 4.10, 'juros_brasil': 13.75, 'pib': 2.0}

    for cod, nome in [(433, 'inflacao'), (432, 'juros_brasil'), (438, 'pib')]:
        sgs_df = get_sgs(cod, nome)
        if not sgs_df.empty:
            df = df.join(sgs_df, how='left')
        else:
            # Garante que a coluna exista mesmo se a API falhar
            df[nome] = fallbacks[nome]

    # Preenchimento de nulos para garantir que o rolling não quebre
    df = df.ffill().bfill()
    
    # 3. AJUSTE ESTATÍSTICO: Acumulado 12 Meses (Seguro contra KeyError)
    if 'inflacao' in df.columns:
        df['inflacao'] = df['inflacao'].rolling(12, min_periods=1).sum()
    if 'pib' in df.columns:
        df['pib'] = df['pib'].rolling(12, min_periods=1).mean()
    
    # Alvos de Retorno
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    
    return df.dropna(subset=['ibov', 'dolar'])

df_full = load_data_master()

# --- Interface ---
st.title("📊 Master Dashboard: Projeção Ibovespa")
st.caption(f"Dados atualizados até: {df_full.index[-1].strftime('%d/%m/%Y')} | Dólar Atual: R$ {df_full['dolar'].iloc[-1]:.3f}")

# Sidebar
st.sidebar.header("🔮 Cenário Futuro (Expectativa 12M)")
features = ['juros_brasil', 'dolar', 'inflacao', 'pib']
u_inputs = []

for f in features:
    # Busca o valor mais recente ou usa o fallback para o input
    default_val = float(df_full[f].iloc[-1]) if f in df_full.columns else 0.0
    val = st.sidebar.number_input(f"Expectativa {f}", value=default_val, format="%.2f")
    u_inputs.append(val)

# --- Processamento Multi-Horizonte ---
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_finais = {}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[target])
        if len(df_h) < 24:
            st.warning(f"Histórico insuficiente para {label}.")
            continue
            
        X, y = df_h[features], df_h[target]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        mdl_f = Ridge(alpha=0.5).fit(X_s, y)
        modelos_finais[label] = (mdl_f, scaler)
        pred_u = mdl_f.predict(scaler.transform([u_inputs]))[0]

        # Gráfico Acumulado
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(f"Performance do Modelo - {label}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y.cumsum(), label="Real", color="black", alpha=0.5)
            # Simulação simples de backtest para visualização
            ax.plot(mdl_f.predict(X_s).cumsum(), label="Ajuste do Modelo", color="#1f77b4", ls="--")
            ax.legend()
            st.pyplot(fig)
        with c2:
            st.metric("Retorno Projetado", f"{pred_u:.2%}")
            st.metric("Ibov Alvo", f"{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")

# --- Sensibilidade ---
st.divider()
st.header("3. Stress Test: Juros vs Dólar (Horizonte 1 Mês)")
if "1 Mês" in modelos_finais:
    mdl_s, scaler_s = modelos_finais["1 Mês"]
    j_base, d_base = u_inputs[0], u_inputs[1]
    
    # Variação de 0.50% Selic e 0.10 Dólar
    j_range = [j_base + x for x in [-1.0, -0.5, 0, 0.5, 1.0]]
    d_range = [d_base + x for x in [-0.2, -0.1, 0, 0.1, 0.2]]
    
    matrix = [[mdl_s.predict(scaler_s.transform([[j, d, u_inputs[2], u_inputs[3]]]))[0] for d in d_range] for j in j_range]
    df_sens = pd.DataFrame(matrix, index=[f"Selic {x:.2f}%" for x in j_range], columns=[f"Dólar R${x:.2f}" for x in d_range])
    
    st.dataframe(df_sens.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=None))

st.divider()
st.header("4. Base de Dados Bruta")
st.dataframe(df_full.tail(15))
