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

st.set_page_config(layout="wide", page_title="Ibov Projeção Real", page_icon="📈")

# --- BOTÃO DE REFRESH ---
if st.sidebar.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

# --- FUNÇÕES DE EXTRAÇÃO COM LOGS TÉCNICOS ---

def get_ibov_strict(start_str, logs):
    """Busca estritamente o fechamento real do Ibovespa (^BVSP)"""
    try:
        # Tenta baixar o ticker oficial
        data = yf.download("^BVSP", start=start_str, progress=False)
        if not data.empty:
            # Tratamento para garantir extração correta independente da versão do yfinance
            df_close = data['Adj Close'].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data['Adj Close']
            return df_close.to_frame('ibov')
        else:
            logs['Yahoo Finance (^BVSP)'] = "Resposta vazia. Possível bloqueio de Rate Limit."
    except Exception as e:
        logs['Yahoo Finance (^BVSP)'] = f"Erro técnico: {str(e)}"
    return pd.DataFrame()

def get_sgs_csv(codigo, nome_coluna, logs):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',')
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df = df.rename(columns={'valor': nome_coluna}).set_index('data')
            return df[[nome_coluna]]
        logs[f'SGS {codigo}'] = f"HTTP {response.status_code}"
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
    logs = {}
    # Defasagem de segurança: 10 anos até 2 dias atrás
    hoje = datetime.now() - timedelta(days=2)
    start_str = (hoje - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # 1. Ibovespa Estrito
    ibov_df = get_ibov_strict(start_str, logs)
    if ibov_df.empty:
        return None, logs
    
    ibov_mensal = ibov_df.resample('ME').last()

    # 2. Dados Macro (D-2)
    dolar = get_sgs_csv(1, 'dolar', logs)
    ipca = get_sgs_csv(433, 'inflacao', logs)
    selic = get_sgs_csv(4390, 'juros_brasil', logs)
    pib = get_sgs_csv(438, 'pib', logs)
    juros_usa = get_fred_csv('FEDFUNDS', 'juros_americano', logs)

    # 3. Consolidação
    df = ibov_mensal.copy()
    for d in [dolar, ipca, selic, pib, juros_usa]:
        if not d.empty:
            df = df.join(d.resample('ME').last(), how='left')

    df = df.ffill().dropna()
    df['target_ret'] = df['ibov'].pct_change().shift(-1)
    
    return df.dropna(), logs

# Execução
data, erros_reais = load_all_data()

# -------------------
# INTERFACE PRINCIPAL
# -------------------

if data is None:
    st.title("📈 Aguardando Conexão ^BVSP")
    st.error("Não foi possível obter a cotação real do Ibovespa agora. Veja o relatório técnico abaixo.")
else:
    st.title("📈 Projeção Ibovespa (^BVSP)")
    st.caption(f"Fechamento real mensal consolidado até: {data.index[-1].strftime('%d/%m/%Y')}")

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
    c1.metric("Retorno Projetado (M+1)", f"{pred_ret:.2%}")
    c2.metric("Ibov Alvo Estimado", f"{data['ibov'].iloc[-1]*(1+pred_ret):,.0f}")
    c3.metric("R² (Aderência Histórica)", f"{model.score(X_scaled, y):.2f}")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Pesos das Variáveis Macro")
        fig, ax = plt.subplots()
        pd.Series(model.coef_, index=features_presentes).sort_values().plot(kind='barh', ax=ax, color='teal')
        st.pyplot(fig)
    with col_r:
        st.subheader("Evolução Histórica Real")
        st.line_chart(data['ibov'])

# --- RELATÓRIO TÉCNICO ---
st.divider()
with st.expander("🛠️ Investigação Técnica (Logs Reais)"):
    if not erros_reais:
        st.success("Todas as APIs responderam com sucesso!")
    else:
        st.info("Abaixo estão os erros retornados pelas APIs durante a última tentativa:")
        for api, erro in erros_reais.items():
            st.code(f"{api}: {erro}")
