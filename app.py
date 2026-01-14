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

# Configuração de página deve ser a primeira instrução Streamlit
st.set_page_config(layout="wide", page_title="Ibov Projeção", page_icon="📈")

# --- FUNÇÕES DE SUPORTE (FORA DO CACHE) ---
def get_sgs_csv(codigo):
    """Busca dados do Banco Central via CSV direto"""
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',')
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df.set_index('data', inplace=True)
            return df['valor']
    except Exception as e:
        st.warning(f"Erro ao acessar série SGS {codigo}: {e}")
    return pd.Series()

def get_fred_csv(series_id):
    """Busca dados do FRED via CSV direto"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url, index_col='DATE', parse_dates=True)
        return df[series_id]
    except Exception as e:
        st.warning(f"Erro ao acessar FRED {series_id}: {e}")
    return pd.Series()

# --- CARREGAMENTO DE DADOS COM CACHE ---
@st.cache_data(ttl=None, show_spinner="📦 Coletando dados macroeconômicos...")
def load_all_data():
    # Janela de 10 anos
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # 1. Ibovespa (Yahoo Finance)
    try:
        ibov_raw = yf.download("^BVSP", start=start_date, progress=False)
        if isinstance(ibov_raw.columns, pd.MultiIndex):
            ibov = ibov_raw['Adj Close'].iloc[:, 0]
        else:
            ibov = ibov_raw['Adj Close']
        ibov = ibov.resample('ME').last()
    except:
        st.error("Falha ao conectar com Yahoo Finance.")
        st.stop()

    # 2. Dados Banco Central (1: Dólar, 433: IPCA, 4390: Selic, 438: PIB)
    dolar = get_sgs_csv(1)
    ipca = get_sgs_csv(433)
    selic = get_sgs_csv(4390)
    pib = get_sgs_csv(438)
    
    # 3. Juros USA (FRED)
    juros_usa = get_fred_csv('FEDFUNDS')

    # 4. Consolidação
    df = pd.DataFrame(index=ibov.index)
    df['ibov'] = ibov
    
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
            # Alinha as datas e preenche buracos
            df = df.join(s.resample('ME').last(), how='left')

    df = df.ffill().dropna()
    # Criar alvo: Retorno do mês seguinte
    df['target_ret'] = df['ibov'].pct_change().shift(-1)
    return df.dropna()

# --- INTERFACE ---
st.title("📈 Projeção Ibovespa: Ridge Regression")

if st.sidebar.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

try:
    data_final = load_all_data()
except Exception as e:
    st.error(f"Erro ao processar dados: {e}")
    st.stop()

# --- MODELO ---
features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = data_final[features]
y = data_final["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Cenário de Simulação")
user_inputs = []
for f in features:
    val = st.sidebar.number_input(f, value=float(X[f].iloc[-1]))
    user_inputs.append(val)

# --- RESULTADOS ---
pred_ret = model.predict(scaler.transform([user_inputs]))[0]

c1, c2, c3 = st.columns(3)
c1.metric("Retorno Projetado (M+1)", f"{pred_ret:.2%}")
c2.metric("Ibov Alvo", f"{data_final['ibov'].iloc[-1]*(1+pred_ret):,.0f}")
c3.metric("R² do Modelo", f"{model.score(X_scaled, y):.2f}")

st.divider()
st.subheader("Análise Gráfica")
col_l, col_r = st.columns(2)

with col_l:
    fig, ax = plt.subplots()
    pd.Series(model.coef_, index=features).sort_values().plot(kind='barh', ax=ax, color='teal')
    ax.set_title("Pesos das Variáveis")
    st.pyplot(fig)

with col_r:
    st.line_chart(data_final['ibov'])
