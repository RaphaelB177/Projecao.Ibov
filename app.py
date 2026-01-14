import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

st.set_page_config(layout="wide", page_title="Ibov Projeção Macro", page_icon="📈")

# --- BOTÃO DE REFRESH ---
if st.sidebar.button("🔄 Forçar Atualização (API Refresh)"):
    st.cache_data.clear()
    st.rerun()

# --- DOWNLOAD YFINANCE (TRATAMENTO DE ERRO) ---
def download_yf_safe(ticker, start_date):
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        if data.empty: return pd.Series()
        if isinstance(data.columns, pd.MultiIndex):
            col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
            return data[col][ticker]
        return data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    except:
        return pd.Series()

# --- DOWNLOAD SGS/BCB (TRATAMENTO DE ERRO "EXPECTED OBJECT") ---
def get_sgs_safe(dict_series, start_date):
    try:
        # Tenta a conexão oficial
        df = sgs.get(dict_series, start=start_date)
        return df
    except Exception as e:
        st.error(f"⚠️ O Banco Central (SGS) está instável ou em manutenção. Detalhe: {str(e)}")
        # Retorna DataFrame vazio para não quebrar o join
        return pd.DataFrame()

# --- DOWNLOAD FRED (VIA URL DIRETA) ---
def get_fred_safe(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    try:
        df = pd.read_csv(url, index_col='DATE', parse_dates=True)
        return df[df.index >= pd.to_datetime(start_date)]
    except:
        return pd.DataFrame()

# -------------------
# CARREGAMENTO COM CACHE E FALLBACK
# -------------------
@st.cache_data(ttl=None)
def load_data():
    # Limite de 10 anos para evitar erro do BCB
    dez_anos_atras = datetime.now() - timedelta(days=365*10)
    start_date = dez_anos_atras.strftime('%Y-%m-%d')
    
    with st.spinner("Conectando às APIs (pode levar alguns segundos)..."):
        # 1. Ibovespa
        ibov = download_yf_safe("^BVSP", start_date)
        if ibov.empty:
            st.warning("⚠️ Yahoo Finance não respondeu. Usando dados históricos em cache (se houver).")
            st.stop()
        ibov = ibov.resample('ME').last()

        # 2. Banco Central (SGS) - Captura o erro "Expected object" aqui
        dict_sgs = {'dolar': 1, 'inflacao': 433, 'juros_brasil': 4390, 'pib': 438}
        df_sgs = get_sgs_safe(dict_sgs, start_date)
        
        if df_sgs.empty:
            st.info("💡 Dica: O sistema do Banco Central costuma oscilar fora do horário comercial ou em picos de acesso.")
            st.stop()

        # 3. FRED
        juros_usa = get_fred_safe('FEDFUNDS', start_date)
        juros_usa.columns = ['juros_americano']

        # Consolidação
        main_df = pd.DataFrame(index=ibov.index)
        main_df['ibov'] = ibov
        
        for d in [df_sgs, juros_usa]:
            if not d.empty:
                d.index = pd.to_datetime(d.index)
                main_df = main_df.join(d.resample('ME').last(), how='left')

        main_df = main_df.ffill().dropna()
        main_df['target_ret'] = main_df['ibov'].pct_change().shift(-1)
        return main_df.dropna()

df = load_data()

# -------------------
# MODELO E UI (DASHBOARD)
# -------------------
st.title("📈 Projeção Ex-Ante Ibovespa")
st.markdown(f"**Base de dados:** {df.index[0].strftime('%Y')} a {df.index[-1].strftime('%Y')}")

features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)

# Sidebar Simulação
st.sidebar.divider()
st.sidebar.subheader("Cenário Futuro")
user_inputs = []
labels = ["Selic (%)", "Dólar (R$)", "PIB (%)", "IPCA (%)", "Juros EUA (%)"]
for i, f in enumerate(features):
    val = st.sidebar.number_input(labels[i], value=float(df[f].iloc[-1]), format="%.2f")
    user_inputs.append(val)

# Predição
pred_ret = model.predict(scaler.transform([user_inputs]))[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Projeção Retorno (M+1)", f"{pred_ret:.2%}")
with c2:
    st.metric("Ibov Alvo", f"{df['ibov'].iloc[-1] * (1 + pred_ret):,.0f}")
with c3:
    st.metric("Aderência (R²)", f"{model.score(X_scaled, y):.2f}")

# Gráficos
st.divider()
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("🎯 Pesos do Modelo")
    fig, ax = plt.subplots()
    pd.Series(model.coef_, index=labels).sort_values().plot(kind='barh', color='teal', ax=ax)
    st.pyplot(fig)
with col_r:
    st.subheader("📊 Evolução Ibovespa")
    st.line_chart(df['ibov'])
