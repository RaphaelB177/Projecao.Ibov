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

# Configurações iniciais
st.set_page_config(layout="wide", page_title="Ibov Projeção Macro", page_icon="📈")

# --- LÓGICA DE REFRESH MANUAL ---
st.sidebar.title("Configurações")
if st.sidebar.button("🔄 Forçar Atualização (API Refresh)"):
    st.cache_data.clear()
    st.rerun()

# --- FUNÇÃO DE DOWNLOAD ROBUSTA ---
def download_yf_safe(ticker, start_date):
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        if data.empty:
            return pd.Series()

        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                return data['Adj Close'][ticker]
            if 'Close' in data.columns.get_level_values(0):
                return data['Close'][ticker]
        else:
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            if 'Close' in data.columns:
                return data['Close']
        return pd.Series()
    except Exception as e:
        st.error(f"❌ Erro no Yahoo Finance ({ticker}): {str(e)}")
        return pd.Series()

# --- FUNÇÃO FRED VIA CSV ---
def get_fred_data(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    try:
        df_fred = pd.read_csv(url, index_col='DATE', parse_dates=True)
        return df_fred[df_fred.index >= pd.to_datetime(start_date)]
    except:
        return pd.DataFrame()

# -------------------
# CARREGAMENTO COM CACHE (Limite de 10 anos)
# -------------------
@st.cache_data(ttl=None)
def load_data():
    # AJUSTE: O BCB limita séries diárias a 10 anos. 
    # Calculamos 10 anos atrás a partir de hoje (ex: 2026 -> 2016)
    dez_anos_atras = datetime.now() - timedelta(days=365*10)
    start_date = dez_anos_atras.strftime('%Y-%m-%d')
    
    with st.spinner(f"📦 Sincronizando dados desde {dez_anos_atras.year} (Limite de 10 anos)..."):
        # 1. Ibovespa
        ibov = download_yf_safe("^BVSP", start_date)
        if ibov.empty:
            st.info("Aguardando liberação do Yahoo Finance (Rate Limit).")
            st.stop()
        ibov = ibov.resample('ME').last()

        # 2. Dados via Banco Central (SGS)
        try:
            # 1: Dólar (Diária), 433: IPCA (Mensal), 4390: SELIC (Mensal), 438: PIB (Trimestral)
            dict_sgs = {'dolar': 1, 'inflacao': 433, 'juros_brasil': 4390, 'pib': 438}
            df_sgs = sgs.get(dict_sgs, start=start_date)
        except Exception as e:
            st.error(f"❌ Erro na API do Banco Central: {str(e)}")
            st.stop()

        # 3. Juros Americanos (FRED)
        juros_usa = get_fred_data('FEDFUNDS', start_date)
        juros_usa.columns = ['juros_americano']

        # Consolidação
        main_df = pd.DataFrame(index=ibov.index)
        main_df['ibov'] = ibov
        
        for d in [df_sgs, juros_usa]:
            d.index = pd.to_datetime(d.index)
            main_df = main_df.join(d.resample('ME').last(), how='left')

        main_df = main_df.ffill().dropna()
        main_df['target_ret'] = main_df['ibov'].pct_change().shift(-1)
        
        return main_df.dropna()

df = load_data()

# -------------------
# MODELAGEM E DASHBOARD (Mantido)
# -------------------
st.title("📈 Projeção Ex-Ante Ibovespa")
st.markdown(f"**Janela de Dados:** {df.index[0].strftime('%m/%Y')} até {df.index[-1].strftime('%m/%Y')} (Máx 10 anos)")

features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)

# Sidebar
st.sidebar.divider()
st.sidebar.subheader("Cenário para Próximo Mês")
user_inputs = []
friendly_names = ["Selic Brasil (%)", "Dólar (R$)", "PIB (Var %)", "IPCA (%)", "Fed Funds (%)"]

for i, f in enumerate(features):
    val = st.sidebar.number_input(friendly_names[i], value=float(df[f].iloc[-1]), format="%.2f")
    user_inputs.append(val)

# Resultados
pred_retorno = model.predict(scaler.transform([user_inputs]))[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Projeção Retorno (1M)", f"{pred_retorno:.2%}")
with c2:
    st.metric("Ibovespa Alvo", f"{df['ibov'].iloc[-1] * (1 + pred_retorno):,.0f}")
with c3:
    st.metric("Aderência (R²)", f"{model.score(X_scaled, y):.2f}")

st.divider()
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("🎯 Importância das Variáveis")
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    pd.Series(model.coef_, index=friendly_names).sort_values().plot(
        kind='barh', color=['#ff4b4b' if x < 0 else '#00cc96' for x in sorted(model.coef_)], ax=ax_imp
    )
    st.pyplot(fig_imp)

with col_right:
    st.subheader("📊 Histórico Ibovespa (10 anos)")
    st.line_chart(df['ibov'])
