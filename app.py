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

st.set_page_config(layout="wide", page_title="Ibov Projeção", page_icon="📈")

# --- BOTÃO DE REFRESH ---
if st.sidebar.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

# --- FUNÇÕES DE EXTRAÇÃO ROBUSTAS ---

def get_sgs_csv(codigo, nome_coluna):
    """Busca dados do BCB e renomeia a coluna imediatamente"""
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',')
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df = df.rename(columns={'valor': nome_coluna})
            df.set_index('data', inplace=True)
            return df[[nome_coluna]]
    except Exception as e:
        st.sidebar.warning(f"SGS {codigo} ({nome_coluna}) offline. Usando último disponível.")
    return pd.DataFrame()

def get_fred_csv(series_id, nome_coluna):
    """Busca dados do FRED e trata erro de índice"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        # O FRED costuma usar 'DATE' ou 'date'. Vamos forçar:
        df.columns = [c.upper() for c in df.columns]
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.rename(columns={series_id: nome_coluna})
            df.set_index('DATE', inplace=True)
            return df[[nome_coluna]]
    except Exception as e:
        st.sidebar.warning(f"FRED {series_id} offline.")
    return pd.DataFrame()

# --- CARREGAMENTO DE DADOS ---

@st.cache_data(ttl=None, show_spinner="📦 Sincronizando indicadores macro...")
def load_all_data():
    # 1. Definir datas (Janela de 10 anos até 2 dias atrás para segurança)
    hoje = datetime.now() - timedelta(days=2)
    dez_anos_atras = hoje - timedelta(days=365*10)
    start_str = dez_anos_atras.strftime('%Y-%m-%d')
    end_str = hoje.strftime('%Y-%m-%d')
    
    # 2. Ibovespa (Yahoo Finance) - Tenta pegar o máximo possível
    try:
        ibov_raw = yf.download("^BVSP", start=start_str, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
        if isinstance(ibov_raw.columns, pd.MultiIndex):
            ibov = ibov_raw['Adj Close'].iloc[:, 0]
        else:
            ibov = ibov_raw['Adj Close']
        ibov_mensal = ibov.resample('ME').last().to_frame('ibov')
    except:
        st.error("Erro fatal: Yahoo Finance não respondeu.")
        st.stop()

    # 3. Extração Individual com Renomeação (Evita Overlap)
    dolar = get_sgs_csv(1, 'dolar')
    ipca = get_sgs_csv(433, 'inflacao')
    selic = get_sgs_csv(4390, 'juros_brasil')
    pib = get_sgs_csv(438, 'pib')
    juros_usa = get_fred_csv('FEDFUNDS', 'juros_americano')

    # 4. Consolidação via Join (Agora sem erro de colunas iguais)
    df = ibov_mensal.copy()
    
    lista_dfs = [dolar, ipca, selic, pib, juros_usa]
    for d in lista_dfs:
        if not d.empty:
            # Resample para mensal para alinhar com Ibov
            d_mensal = d.resample('ME').last()
            df = df.join(d_mensal, how='left')

    # Tratamento final: preenche buracos e remove nulos iniciais
    df = df.ffill().dropna()
    
    # Target: Retorno do mês seguinte
    df['target_ret'] = df['ibov'].pct_change().shift(-1)
    return df.dropna()

# --- EXECUÇÃO ---

data = load_all_data()

st.title("📈 Projeção Ibovespa (Defasagem de Segurança)")
st.info(f"Dados sincronizados até: {data.index[-1].strftime('%d/%m/%Y')} (D-2 aplicado)")

# --- MODELO RIDGE ---
features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
# Verifica se todas as colunas existem antes de treinar
features_presentes = [f for f in features if f in data.columns]

X = data[features_presentes]
y = data["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)

# --- SIDEBAR ---
st.sidebar.header("Simular Próximo Mês")
user_inputs = []
for f in features_presentes:
    val = st.sidebar.number_input(f"Valor para {f}", value=float(X[f].iloc[-1]), format="%.2f")
    user_inputs.append(val)

# --- RESULTADOS ---
pred_ret = model.predict(scaler.transform([user_inputs]))[0]

c1, c2, c3 = st.columns(3)
c1.metric("Retorno Projetado", f"{pred_ret:.2%}")
c2.metric("Ibov Alvo", f"{data['ibov'].iloc[-1]*(1+pred_ret):,.0f}")
c3.metric("R² do Modelo", f"{model.score(X_scaled, y):.2f}")

# --- GRÁFICOS ---
st.divider()
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Pesos do Modelo")
    fig, ax = plt.subplots()
    pd.Series(model.coef_, index=features_presentes).sort_values().plot(kind='barh', ax=ax, color='teal')
    st.pyplot(fig)

with col_r:
    st.subheader("Evolução Ibovespa")
    st.line_chart(data['ibov'])
