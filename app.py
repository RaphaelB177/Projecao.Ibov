import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import requests

st.set_page_config(layout="wide", page_title="Ibov Projeção Macro", page_icon="📈")

# --- BOTÃO DE REFRESH ---
if st.sidebar.button("🔄 Forçar Atualização (API Refresh)"):
    st.cache_data.clear()
    st.rerun()

# --- FUNÇÃO DOWNLOAD YFINANCE ---
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

# --- FUNÇÃO SGS VIA CSV (MAIS ESTÁVEL QUE JSON) ---
def get_sgs_csv(codigo):
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # O BCB usa ";" como separador e "," para decimais no CSV
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',')
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df.set_index('data', inplace=True)
            return df['valor']
        return pd.Series()
    except:
        return pd.Series()

# --- FUNÇÃO FRED ---
def get_fred_safe(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    try:
        df = pd.read_csv(url, index_col='DATE', parse_dates=True)
        return df[df.index >= pd.to_datetime(start_date)]
    except:
        return pd.DataFrame()

# -------------------
# CARREGAMENTO DE DADOS
# -------------------
@st.cache_data(ttl=None)
def load_data():
    dez_anos_atras = datetime.now() - timedelta(days=365*10)
    start_date = dez_anos_atras.strftime('%Y-%m-%d')
    
    with st.spinner("📦 Sincronizando com servidores do BCB e Yahoo..."):
        # 1. Ibovespa
        ibov = download_yf_safe("^BVSP", start_date)
        if ibov.empty:
            st.warning("⚠️ Yahoo Finance indisponível no momento.")
            st.stop()
        ibov = ibov.resample('ME').last()

        # 2. Dados Banco Central (Via CSV Direto)
        # 1: Dólar, 433: IPCA, 4390: SELIC, 438: PIB
        dolar = get_sgs_csv(1)
        ipca = get_sgs_csv(433)
        selic = get_sgs_csv(4390)
        pib = get_sgs_csv(438)
        
        # 3. FRED
        juros_usa = get_fred_safe('FEDFUNDS', start_date)

        # Consolidação
        main_df = pd.DataFrame(index=ibov.index)
        main_df['ibov'] = ibov
        
        # Join das séries (com verificação se não estão vazias)
        series_map = {
            'dolar': dolar, 
            'inflacao': ipca, 
            'juros_brasil': selic, 
            'pib': pib,
            'juros_americano': juros_usa
        }
        
        for name, s in series_map.items():
            if not s.empty:
                s.index = pd.to_datetime(s.index)
                main_df = main_df.join(s.resample('ME').last(), how='left')

        main_df = main_df.ffill().dropna()
        main_df['target_ret'] = main_df['ibov'].pct_change().shift(-1)
        return main_df.dropna()

df = load_data()

# -------------------
# MODELO E DASHBOARD
# -------------------
st.title("📈 Projeção Ex-Ante Ibovespa")
st.markdown(f"**Janela de Dados:** {df.index[0].strftime('%m/%Y')} a {df.index[-1].strftime('%m/%Y')}")

features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)

# Sidebar Simulação
st.sidebar.divider()
st.sidebar.subheader("Cenário de Simulação")
user_inputs = []
labels = ["Selic (%)", "Dólar (R$)", "PIB (%)", "IPCA (%)", "Juros EUA (%)"]
for i, f in enumerate(features):
    val = st.sidebar.number_input(labels[i], value=float(df[f].iloc[-1]), format="%.2f")
    user_inputs.append(val)

# Predição
pred_ret = model.predict(scaler.transform([user_inputs]))[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Retorno Projetado (M+1)", f"{pred_ret:.2%}")
with c2:
    st.metric("Ibov Alvo", f"{df['ibov'].iloc[-1] * (1 + pred_ret):,.0f}")
with c3:
    st.metric("R² do Modelo", f"{model.score(X_scaled, y):.2f}")

st.divider()
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("🎯 Pesos das Variáveis")
    fig, ax = plt.subplots()
    pd.Series(model.coef_, index=labels).sort_values().plot(kind='barh', color='teal', ax=ax)
    st.pyplot(fig)
with col_r:
    st.subheader("📊 Histórico Ibovespa")
    st.line_chart(df['ibov'])
