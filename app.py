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

# --- BOTÃO DE REFRESH NO TOPO DA SIDEBAR ---
if st.sidebar.button("🔄 Atualizar Dados das APIs"):
    st.cache_data.clear() # Limpa o cache, forçando o download na próxima execução
    st.rerun()

# --- Função de Download com Retry ---
def download_yf_with_retry(ticker, start_date, retries=2):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    return data['Adj Close'][ticker]
                return data['Adj Close']
        except:
            time.sleep(1)
    return pd.Series()

# --- Função para baixar FRED ---
def get_fred_data(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    try:
        return pd.read_csv(url, index_col='DATE', parse_dates=True)
    except:
        return pd.DataFrame()

# -------------------
# Carregamento de Dados COM CACHE LONGO
# -------------------
@st.cache_data(ttl=None) # TTL=None faz com que os dados nunca expirem sozinhos
def load_data():
    start_date = "2010-01-01"
    
    # Aviso visual de que o app está buscando dados reais
    with st.spinner("Buscando novos dados das APIs (Yahoo, BCB, FRED)..."):
        # 1. Ibovespa
        ibov = download_yf_with_retry("^BVSP", start_date)
        if ibov.empty:
            st.error("Erro no Yahoo Finance. Tente o Refresh novamente em instantes.")
            st.stop()
        ibov = ibov.resample('ME').last()

        # 2. Dados via Banco Central (SGS)
        try:
            dict_sgs = {'dolar': 1, 'inflacao': 433, 'juros_brasil': 4390, 'pib': 438}
            df_sgs = sgs.get(dict_sgs, start=start_date)
        except:
            st.error("Erro no Banco Central.")
            st.stop()

        # 3. Juros USA
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

# Chama a função de dados
df = load_data()

# -------------------
# Interface e Modelo
# -------------------
st.title("📈 Projeção Ibovespa (Ex-Ante)")
st.info(f"Dados em cache desde o último refresh. Última data disponível: {df.index[-1].strftime('%d/%m/%Y')}")

# O restante do seu código de modelo e gráficos continua aqui...
# [Ridge Regression, Scaler, Inputs do Usuário e Plots]

features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)

# Sidebar para inputs
st.sidebar.divider()
st.sidebar.header("Cenário de Projeção")
user_vals = []
input_names = ["Selic (%)", "Dólar (R$)", "PIB (%)", "IPCA (%)", "Juros EUA (%)"]
for i, f in enumerate(features):
    val = st.sidebar.number_input(input_names[i], value=float(df[f].iloc[-1]))
    user_vals.append(val)

# Predição e Plots
pred_ret = model.predict(scaler.transform([user_vals]))[0]

c1, c2 = st.columns(2)
with c1:
    st.metric("Projeção Retorno (Próximo Mês)", f"{pred_ret:.2%}")
    st.metric("Ibov Alvo", f"{df['ibov'].iloc[-1] * (1+pred_ret):,.0f}")
with c2:
    fig, ax = plt.subplots()
    pd.Series(model.coef_, index=input_names).plot(kind='barh', ax=ax, color='teal')
    st.pyplot(fig)
