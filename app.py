import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide", page_title="Ibov Projeção Macro")

# --- LÓGICA DE REFRESH ---
# O botão limpa o cache e recarrega a página, forçando uma nova consulta às APIs
if st.sidebar.button("🔄 Forçar Atualização (API Refresh)"):
    st.cache_data.clear()
    st.rerun()

# --- FUNÇÃO DE DOWNLOAD COM DEBUG ---
def download_yf_debug(ticker, start_date):
    try:
        # Tentamos baixar o dado
        data = yf.download(ticker, start=start_date, progress=False)
        
        if data.empty:
            st.error(f"⚠️ Yahoo retornou DataFrame vazio para {ticker}. Pode ser um bloqueio de IP ou ticker inválido.")
            return pd.Series()
            
        # Tratamento para garantir a extração da coluna 'Adj Close'
        if isinstance(data.columns, pd.MultiIndex):
            return data['Adj Close'][ticker]
        return data['Adj Close']
        
    except Exception as e:
        # EXIBE O ERRO REAL PARA INVESTIGAÇÃO
        st.error(f"❌ Erro Crítico no Yahoo Finance ({ticker}): {type(e).__name__} - {str(e)}")
        return pd.Series()

# --- FUNÇÃO FRED VIA CSV ---
def get_fred_data(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    try:
        df_fred = pd.read_csv(url, index_col='DATE', parse_dates=True)
        return df_fred[df_fred.index >= start_date]
    except Exception as e:
        st.warning(f"⚠️ Erro ao buscar dados do FRED: {str(e)}")
        return pd.DataFrame()

# -------------------
# CARREGAMENTO COM CACHE PERSISTENTE
# -------------------
@st.cache_data(ttl=None) # Só atualiza se clicar no botão de Refresh
def load_data():
    start_date = "2010-01-01"
    
    with st.spinner("Consultando APIs externas..."):
        # 1. Ibovespa via Yahoo
        ibov = download_yf_debug("^BVSP", start_date)
        if ibov.empty:
            st.stop() # Interrompe a execução se o dado essencial falhar
        ibov = ibov.resample('ME').last()

        # 2. Dados via Banco Central (SGS)
        try:
            # Usando bcb sgs para Dólar (1), IPCA (433), SELIC (4390), PIB (438)
            dict_sgs = {'dolar': 1, 'inflacao': 433, 'juros_brasil': 4390, 'pib': 438}
            df_sgs = sgs.get(dict_sgs, start=start_date)
        except Exception as e:
            st.error(f"❌ Erro no Banco Central (SGS): {str(e)}")
            st.stop()

        # 3. Juros USA via FRED
        juros_usa = get_fred_data('FEDFUNDS', start_date)
        juros_usa.columns = ['juros_americano']

        # Consolidação Final
        main_df = pd.DataFrame(index=ibov.index)
        main_df['ibov'] = ibov
        
        for d in [df_sgs, juros_usa]:
            d.index = pd.to_datetime(d.index)
            main_df = main_df.join(d.resample('ME').last(), how='left')

        main_df = main_df.ffill().dropna()
        main_df['target_ret'] = main_df['ibov'].pct_change().shift(-1)
        return main_df.dropna()

# Executa a carga
df = load_data()

# -------------------
# MODELAGEM E UI
# -------------------
st.title("📈 Projeção Ibovespa (Ex-Ante)")
st.info(f"Dados em cache. Último fechamento: {df.index[-1].strftime('%d/%m/%Y')}")

features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)

# Sidebar de Cenários
st.sidebar.divider()
st.sidebar.subheader("Simulação de Cenário")
user_inputs = []
labels = ["Selic", "Dólar", "PIB", "IPCA", "Fed Funds"]
for i, f in enumerate(features):
    val = st.sidebar.number_input(f"{labels[i]} atual/esperado", value=float(df[f].iloc[-1]), format="%.2f")
    user_inputs.append(val)

# Resultado
pred_ret = model.predict(scaler.transform([user_inputs]))[0]

c1, c2 = st.columns(2)
with c1:
    st.metric("Retorno Projetado (Mês+1)", f"{pred_ret:.2%}")
    st.metric("Ibov Alvo Estimado", f"{df['ibov'].iloc[-1] * (1+pred_ret):,.0f}")
with c2:
    st.subheader("Importância das Variáveis")
    fig, ax = plt.subplots()
    pd.Series(model.coef_, index=labels).plot(kind='barh', ax=ax, color='teal')
    st.pyplot(fig)
