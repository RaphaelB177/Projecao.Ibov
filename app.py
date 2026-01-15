import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import requests

st.set_page_config(layout="wide", page_title="Master Ibov Dashboard", page_icon="📈")

@st.cache_data(ttl=3600)
def load_data_master():
    hoje = datetime.now()
    start_str = (hoje - timedelta(days=365*15)).strftime('%Y-%m-%d')
    
    # 1. Ibov e Dólar (Yahoo Finance) - Puxando o histórico REAL
    try:
        data_yf = yf.download(["^BVSP", "BRL=X"], start=start_str, progress=False)
        if isinstance(data_yf.columns, pd.MultiIndex):
            ibov = data_yf['Adj Close']['^BVSP']
            dolar = data_yf['Adj Close']['BRL=X']
        else:
            ibov = data_yf['^BVSP']
            dolar = data_yf['BRL=X']
        df = pd.DataFrame({'ibov': ibov, 'dolar': dolar})
    except:
        st.error("Erro ao baixar histórico do Yahoo. Verifique a conexão.")
        st.stop()

    # 2. Séries do Banco Central (SGS) - Histórico REAL
    def get_sgs(codigo, nome):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
        try:
            r = requests.get(url, timeout=10)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            return d.rename(columns={'valor': nome}).set_index('data')
        except: return pd.DataFrame()

    # Pegando Juros (432), Inflação (433) e PIB (438)
    for cod, nome in [(433, 'inflacao'), (432, 'juros_brasil'), (438, 'pib')]:
        sgs_df = get_sgs(cod, nome)
        if not sgs_df.empty:
            df = df.join(sgs_df, how='left')

    # RESAMPLING MENSAL (Para alinhar as séries e evitar quebras)
    df = df.resample('ME').last()

    # PREENCHIMENTO INTELIGENTE: 
    # Primeiro ffill (repete o último real), depois bfill (para o início da série)
    # Isso evita que o histórico fique "estático" com um valor só
    df = df.ffill().bfill()
    
    # 3. AJUSTE ESTATÍSTICO: Acumulado 12 Meses
    # Só aplicamos o rolling se a coluna tiver dados variados
    df['inflacao'] = df['inflacao'].rolling(12, min_periods=6).sum()
    df['pib'] = df['pib'].rolling(12, min_periods=6).mean()
    
    # Horizontes de Retorno
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    
    return df.dropna(subset=['ibov', 'dolar', 'inflacao'])

df_full = load_data_master()

# --- Interface ---
st.title("📊 Master Dashboard: Projeção Ibovespa")
st.caption(f"Dados Reais de {df_full.index[0].year} até {df_full.index[-1].strftime('%d/%m/%Y')}")

# Sidebar
st.sidebar.header("🔮 Cenário Futuro (Expectativa 12M)")
features = ['juros_brasil', 'dolar', 'inflacao', 'pib']
u_inputs = []

for f in features:
    # O valor padrão é o ÚLTIMO dado real da série histórica
    val = st.sidebar.number_input(f"Expectativa {f}", value=float(df_full[f].iloc[-1]), format="%.2f")
    u_inputs.append(val)

# --- Processamento ---
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_finais = {}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[target])
        X, y = df_h[features], df_h[target]
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        mdl_f = Ridge(alpha=0.5).fit(X_s, y)
        modelos_finais[label] = (mdl_f, scaler)
        pred_u = mdl_f.predict(scaler.transform([u_inputs]))[0]

        col1, col2 = st.columns([2, 1])
        with col1:
            # Gráfico do HISTÓRICO REAL vs AJUSTE DO MODELO
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y.index, y.cumsum(), label="Histórico Real (Acumulado)", color="black", alpha=0.4)
            ax.plot(y.index, mdl_f.predict(X_s).cumsum(), label="Ajuste Estatístico", color="#1f77b4")
            ax.set_title(f"Aderência ao Histórico - {label}")
            ax.legend()
            st.pyplot(fig)
        with col2:
            st.metric("Retorno Projetado", f"{pred_u:.2%}")
            st.metric("Ibov Alvo", f"{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")

# --- Sensibilidade ---
st.divider()
st.header("3. Stress Test: Juros vs Dólar (Horizonte 1 Mês)")
if "1 Mês" in modelos_finais:
    mdl_s, scaler_s = modelos_finais["1 Mês"]
    j_base, d_base = u_inputs[0], u_inputs[1]
    
    # Variações de 0.50% Selic e 0.10 Dólar
    j_range = [j_base + x for x in [-1.0, -0.5, 0, 0.5, 1.0]]
    d_range = [d_base + x for x in [-0.2, -0.1, 0, 0.1, 0.2]]
    
    matrix = [[mdl_s.predict(scaler_s.transform([[j, d, u_inputs[2], u_inputs[3]]]))[0] for d in d_range] for j in j_range]
    df_sens = pd.DataFrame(matrix, index=[f"Selic {x:.2f}%" for x in j_range], columns=[f"Dólar R${x:.2f}" for x in d_range])
    st.dataframe(df_sens.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=None))

st.divider()
st.header("4. Base de Dados Bruta (Histórico Real)")
st.dataframe(df_full.tail(15))
