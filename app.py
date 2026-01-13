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

# --- Função de Download com Retry para contornar Rate Limit ---
def download_yf_with_retry(ticker, start_date, retries=3):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, progress=False)
            if not data.empty:
                # Ajuste para garantir que pegamos a coluna correta em multi-index ou single
                if isinstance(data.columns, pd.MultiIndex):
                    return data['Adj Close'][ticker]
                return data['Adj Close']
        except Exception:
            time.sleep(2) # Espera 2 segundos antes de tentar de novo
    return pd.Series()

# --- Função para baixar FRED via CSV direto ---
def get_fred_data(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    try:
        df_fred = pd.read_csv(url, index_col='DATE', parse_dates=True)
        return df_fred[df_fred.index >= start_date]
    except:
        return pd.DataFrame()

# -------------------
# Carregamento de Dados
# -------------------
@st.cache_data(ttl=3600)
def load_data():
    start_date = "2010-01-01"
    
    # 1. Ibovespa (Yahoo) - Com Retry
    ibov = download_yf_with_retry("^BVSP", start_date)
    if ibov.empty:
        st.error("O Yahoo Finance bloqueou a requisição temporariamente (Rate Limit). Tente atualizar a página em alguns instantes.")
        st.stop()
    ibov = ibov.resample('ME').last()

    # 2. Dados via Banco Central (BCB/SGS) - Mais estável que o Yahoo
    # 1: Dólar Venda, 433: IPCA, 4390: SELIC, 438: PIB
    try:
        dict_sgs = {
            'dolar': 1,
            'inflacao': 433,
            'juros_brasil': 4390,
            'pib': 438
        }
        df_sgs = sgs.get(dict_sgs, start=start_date)
    except Exception as e:
        st.error(f"Erro ao conectar com o Banco Central: {e}")
        st.stop()

    # 3. Juros USA via FRED
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

# --- Restante do Modelo (Ridge, Scaler, Plots) permanece igual ---
st.title("📈 Projeção Ibovespa (Ex-Ante)")
st.write(f"Dados atualizados até: {df.index[-1].strftime('%d/%m/%Y')}")

features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=1.0).fit(X_scaled, y)

# Sidebar amigável
st.sidebar.header("Configure o Cenário")
input_map = {
    "juros_brasil": "Selic Brasil (%)",
    "dolar": "Câmbio (R$)",
    "pib": "Crescimento PIB (%)",
    "inflacao": "IPCA (%)",
    "juros_americano": "Juros EUA (%)"
}

user_vals = []
for f in features:
    val = st.sidebar.number_input(input_map[f], value=float(df[f].iloc[-1]), format="%.2f")
    user_vals.append(val)

# Predição
user_scaled = scaler.transform([user_vals])
pred_ret = model.predict(user_scaled)[0]

c1, c2 = st.columns(2)
with c1:
    st.metric("Projeção Retorno (Próximo Mês)", f"{pred_ret:.2%}")
    st.metric("Ibov Alvo", f"{df['ibov'].iloc[-1] * (1+pred_ret):,.0f}")

with c2:
    st.subheader("Peso das Variáveis")
    fig, ax = plt.subplots()
    colors = ['red' if x < 0 else 'green' for x in model.coef_]
    pd.Series(model.coef_, index=[input_map[f] for f in features]).plot(kind='barh', ax=ax, color=colors)
    st.pyplot(fig)
