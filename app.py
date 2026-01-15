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
import time

st.set_page_config(layout="wide", page_title="Ibov Projeção", page_icon="📈")

# --- BOTÃO DE REFRESH ---
if st.sidebar.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

# --- FUNÇÕES DE EXTRAÇÃO COM LOG SILENCIOSO ---

def get_ibov_data(start_str, logs):
    """Tenta baixar Ibovespa; se falhar, tenta EWZ."""
    tickers = ["^BVSP", "EWZ"]
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_str, progress=False)
            if not data.empty:
                df_close = data['Adj Close'].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data['Adj Close']
                return df_close.to_frame('ibov'), ticker
        except Exception as e:
            logs['Yahoo Finance'] = f"Erro no ticker {ticker}: {str(e)}"
    return pd.DataFrame(), None

def get_sgs_csv(codigo, nome_coluna, logs):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',')
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df = df.rename(columns={'valor': nome_coluna}).set_index('data')
            return df[[nome_coluna]]
        else:
            logs[f'SGS {codigo}'] = f"Status Code: {response.status_code}"
    except Exception as e:
        logs[f'SGS {codigo}'] = str(e)
    return pd.DataFrame()

def get_fred_csv(series_id, nome_coluna, logs):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        df.columns = [c.upper() for c in df.columns]
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.rename(columns={series_id: nome_coluna}).set_index('DATE')
            return df[[nome_coluna]]
    except Exception as e:
        logs[f'FRED {series_id}'] = str(e)
    return pd.DataFrame()

# --- CARREGAMENTO CENTRAL ---

@st.cache_data(ttl=None)
def load_all_data():
    logs = {} # Dicionário para capturar erros sem exibir avisos
    hoje = datetime.now() - timedelta(days=2)
    start_str = (hoje - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # 1. Ibovespa
    ibov_df, ticker_usado = get_ibov_data(start_str, logs)
    if ibov_df.empty:
        return None, None, logs
    
    ibov_mensal = ibov_df.resample('ME').last()

    # 2. Dados Macro
    dolar = get_sgs_csv(1, 'dolar', logs)
    ipca = get_sgs_csv(433, 'inflacao', logs)
    selic = get_sgs_csv(4390, 'juros_brasil', logs)
    pib = get_sgs_csv(438, 'pib', logs)
    juros_usa = get_fred_csv('FEDFUNDS', 'juros_americano', logs)

    # 3. Join
    df = ibov_mensal.copy()
    for d in [dolar, ipca, selic, pib, juros_usa]:
        if not d.empty:
            df = df.join(d.resample('ME').last(), how='left')

    df = df.ffill().dropna()
    df['target_ret'] = df['ibov'].pct_change().shift(-1)
    
    return df.dropna(), ticker_label_map(ticker_usado), logs

def ticker_label_map(t):
    return "Ibovespa (^BVSP)" if t == "^BVSP" else "Proxy Brasil (EWZ)"

# Execução
data, ticker_info, erros_reais = load_all_data()

# -------------------
# DASHBOARD
# -------------------

if data is None:
    st.title("📈 Sistema Temporariamente Indisponível")
    st.warning("Não foi possível carregar o Ibovespa. Verifique o relatório de erros abaixo.")
else:
    st.title("📈 Projeção Ibovespa")
    st.caption(f"Dados via {ticker_info} | Sincronizado até: {data.index[-1].strftime('%d/%m/%Y')}")

    # Modelo Ridge
    features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
    features_presentes = [f for f in features if f in data.columns]
    
    X = data[features_presentes]
    y = data["target_ret"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0).fit(X_scaled, y)

    # Sidebar Simulação
    st.sidebar.divider()
    st.sidebar.header("Cenário Simulado")
    user_inputs = []
    for f in features_presentes:
        val = st.sidebar.number_input(f, value=float(X[f].iloc[-1]))
        user_inputs.append(val)

    # Métricas
    pred_ret = model.predict(scaler.transform([user_inputs]))[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Retorno Projetado", f"{pred_ret:.2%}")
    c2.metric("Ibov Alvo", f"{data['ibov'].iloc[-1]*(1+pred_ret):,.0f}")
    c3.metric("Aderência (R²)", f"{model.score(X_scaled, y):.2f}")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        fig, ax = plt.subplots()
        pd.Series(model.coef_, index=features_presentes).sort_values().plot(kind='barh', ax=ax, color='teal')
        st.pyplot(fig)
    with col_r:
        st.line_chart(data['ibov'])

# --- ÁREA DE DEBUG (ESCONDIDA) ---
st.divider()
with st.expander("🛠️ Relatório Técnico de Erros (Investigação)"):
    if not erros_reais:
        st.success("Nenhum erro técnico detectado nas últimas requisições!")
    else:
        st.write("Abaixo estão os erros reais retornados pelas APIs:")
        for api, erro in erros_reais.items():
            st.error(f"**{api}**: {erro}")
