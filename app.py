import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import requests

# -------------------
# Configurações Iniciais
# -------------------
st.set_page_config(layout="wide", page_title="Dashboard Macro Ibovespa", page_icon="📊")

@st.cache_data(ttl=None)
def load_data_v5():
    hoje = datetime.now() - timedelta(days=2)
    start_str = (hoje - timedelta(days=365*12)).strftime('%Y-%m-%d')
    
    # 1. Ibovespa (Tratamento Multi-index)
    try:
        ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
        if isinstance(ibov_raw.columns, pd.MultiIndex):
            nivel_0 = ibov_raw.columns.get_level_values(0)
            col = 'Adj Close' if 'Adj Close' in nivel_0 else 'Close'
            ibov = ibov_raw[col].iloc[:, 0]
        else:
            col = 'Adj Close' if 'Adj Close' in ibov_raw.columns else 'Close'
            ibov = ibov_raw[col]
        df = ibov.resample('ME').last().to_frame('ibov')
    except:
        st.error("Erro ao carregar Yahoo Finance.")
        st.stop()

    # 2. Função SGS Segura
    def get_sgs_safe(codigo, nome):
        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
            r = requests.get(url, timeout=15)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            d = d.rename(columns={'valor': nome}).set_index('data')
            return d.resample('ME').last()
        except: return pd.DataFrame()

    # Join das Séries
    for cod, nome in [(1, 'dolar'), (433, 'inflacao'), (4390, 'juros_brasil'), (438, 'pib')]:
        m_data = get_sgs_safe(cod, nome)
        if not m_data.empty: df = df.join(m_data, how='left')
    
    df = df.ffill().dropna()
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    return df

df_full = load_data_v5()

# -------------------
# Interface Principal
# -------------------
st.title("📊 Inteligência Estatística Ibovespa")

# Identificar colunas presentes para evitar KeyError
features_all = ["juros_brasil", "dolar", "pib", "inflacao"]
features_presentes = [f for f in features_all if f in df_full.columns]

st.sidebar.header("Cenário de Projeção")
u_inputs = []
for f in features_presentes:
    val = st.sidebar.number_input(f"Expectativa {f}", value=float(df_full[f].iloc[-1]), step=0.01)
    u_inputs.append(val)

# 1. Mapa de Calor de Correlação
st.header("1. Mapa de Calor (Correlação Macro)")
corr_cols = features_presentes + ['ret_1m', 'ret_6m', 'ret_12m']
corr_matrix = df_full[corr_cols].corr().loc[features_presentes, ['ret_1m', 'ret_6m', 'ret_12m']]


fig_corr, ax_corr = plt.subplots(figsize=(12, 3.5))
sns.heatmap(corr_matrix.T, annot=True, cmap="RdYlGn", center=0, ax=ax_corr, fmt=".2f")
st.pyplot(fig_corr)

# 2. Projeções e Backtest
st.header("2. Projeções por Horizonte")
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_treinados = {}

tabs = st.tabs(list(horizontes.keys()))
for i, (label, col_target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[col_target])
        X, y = df_h[features_presentes], df_h[col_target]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Modelo Final
        mdl = Ridge(alpha=0.5).fit(X_s, y)
        modelos_treinados[label] = (mdl, scaler)
        pred = mdl.predict(scaler.transform([u_inputs]))[0]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            # Backtest histórico simplificado
            preds_bt = []
            for j in range(48, len(y)):
                m_bt = Ridge(alpha=0.5).fit(X_s[:j], y[:j])
                preds_bt.append(m_bt.predict(X_s[j:j+1])[0])
            res_bt = pd.Series(preds_bt, index=y.index[48:])
            
            fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
            ax_bt.plot(y[48:].cumsum(), label="Real", color="black", alpha=0.6)
            ax_bt.plot(res_bt.cumsum(), label="Modelo", color="#1f77b4", ls="--")
            ax_bt.legend()
            st.pyplot(fig_bt)
        with c2:
            st.metric("Retorno Projetado", f"{pred:.2%}")
            st.metric("Alvo Estimado", f"{df_full['ibov'].iloc[-1]*(1+pred):,.0f}")

# 3. Tabela de Sensibilidade (Garantida)
st.divider()
st.header("3. Tabela de Sensibilidade (Stress Test)")
st.write("Variação do Retorno Projetado (1 Mês) ao cruzar Juros e Dólar:")

if "juros_brasil" in features_presentes and "dolar" in features_presentes:
    j_idx = features_presentes.index("juros_brasil")
    d_idx = features_presentes.index("dolar")
    
    # Criar eixos (Var de +/- 10% do valor de input)
    j_vals = np.linspace(u_inputs[j_idx]*0.8, u_inputs[j_idx]*1.2, 7)
    d_vals = np.linspace(u_inputs[d_idx]*0.8, u_inputs[d_idx]*1.2, 7)
    
    sens_data = []
    mdl_s, scaler_s = modelos_treinados["1 Mês"]
    
    for j_val in j_vals:
        linha = []
        for d_val in d_vals:
            scen = list(u_inputs)
            scen[j_idx], scen[d_idx] = j_val, d_val
            linha.append(mdl_s.predict(scaler_s.transform([scen]))[0])
        sens_data.append(linha)
    
    df_sens = pd.DataFrame(
        sens_data, 
        index=[f"Selic {x:.2f}%" for x in j_vals],
        columns=[f"Dólar R${x:.2f}" for x in d_vals]
    )

    
    st.dataframe(df_sens.style.format("{:.2%}")
                 .background_gradient(cmap="RdYlGn", axis=None))
else:
    st.warning("Variáveis 'juros_brasil' ou 'dolar' não encontradas para gerar a tabela de sensibilidade.")

# Footer técnico
with st.expander("Base de Dados Bruta"):
    st.write(df_full.tail(10))
