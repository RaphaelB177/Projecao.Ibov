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
# Configurações e Coleta
# -------------------
st.set_page_config(layout="wide", page_title="Ibov Intelligence Dashboard", page_icon="📊")

@st.cache_data(ttl=None)
def load_data_v4():
    hoje = datetime.now() - timedelta(days=2)
    start_str = (hoje - timedelta(days=365*12)).strftime('%Y-%m-%d')
    
    # 1. Ibov (Tratamento Multi-index)
    ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
    if ibov_raw.empty: st.stop()
    
    if isinstance(ibov_raw.columns, pd.MultiIndex):
        nivel_0 = ibov_raw.columns.get_level_values(0)
        col = 'Adj Close' if 'Adj Close' in nivel_0 else 'Close'
        ibov = ibov_raw[col].iloc[:, 0]
    else:
        col = 'Adj Close' if 'Adj Close' in ibov_raw.columns else 'Close'
        ibov = ibov_raw[col]

    df = ibov.resample('ME').last().to_frame('ibov')

    # 2. Função SGS Segura
    def get_sgs_safe(codigo, nome):
        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
            r = requests.get(url, timeout=15)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            d = d.rename(columns={'valor': nome}).set_index('data')
            return d.resample('ME').last() if not d.empty else pd.DataFrame()
        except: return pd.DataFrame()

    # Join Macro
    for cod, nome in [(1, 'dolar'), (433, 'inflacao'), (4390, 'juros_brasil'), (438, 'pib')]:
        m_data = get_sgs_safe(cod, nome)
        if not m_data.empty: df = df.join(m_data, how='left')
    
    df = df.ffill().dropna()
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    return df

df_full = load_data_v4()

# -------------------
# Interface Principal
# -------------------
st.title("📊 Ibovespa: Inteligência Macro e Sensibilidade")

# Sidebar
st.sidebar.header("Cenário e Modelo")
window_type = st.sidebar.selectbox("Janela Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Janela Móvel", 12, 60, 36)

features = ["juros_brasil", "dolar", "pib", "inflacao"]
u_inputs = []
for f in features:
    val = st.sidebar.number_input(f"Expectativa {f}", value=float(df_full[f].iloc[-1]), step=0.01)
    u_inputs.append(val)

# -------------------
# 1. Mapa de Calor (Heatmap)
# -------------------
st.header("1. Força dos Indicadores (Correlação)")
corr_cols = features + ['ret_1m', 'ret_6m', 'ret_12m']
corr_matrix = df_full[corr_cols].corr()
# Filtramos apenas a relação dos indicadores com os retornos futuros
corr_target = corr_matrix.loc[features, ['ret_1m', 'ret_6m', 'ret_12m']]

fig_corr, ax_corr = plt.subplots(figsize=(10, 3))
sns.heatmap(corr_target.T, annot=True, cmap="RdYlGn", center=0, ax=ax_corr)
ax_corr.set_title("Correlação: Variável hoje vs Retorno Ibovespa futuro")
st.pyplot(fig_corr)

# -------------------
# 2. Projeções e Backtest (Tabs)
# -------------------
st.header("2. Projeções por Horizonte")
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_treinados = {} # Guardar para a sensibilidade

tabs = st.tabs(list(horizontes.keys()))

for i, (label, col_target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[col_target])
        X, y = df_h[features], df_h[col_target]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Backtest Simples
        start_idx = 48
        preds_bt, actuals_bt = [], []
        for j in range(start_idx, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            mdl = Ridge(alpha=0.5).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        res_bt = pd.DataFrame({"Real": actuals_bt, "Prev": preds_bt}, index=y.index[start_idx:])
        
        # Projeção Atual
        mdl_f = Ridge(alpha=0.5).fit(X_s, y)
        modelos_treinados[label] = (mdl_f, scaler) # Salva para uso posterior
        pred_u = mdl_f.predict(scaler.transform([u_inputs]))[0]
        
        # Layout
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_h, ax_h = plt.subplots(figsize=(10, 4))
            ax_h.plot(res_bt["Real"].cumsum(), label="Real", color="black")
            ax_h.plot(res_bt["Prev"].cumsum(), label="Previsto", color="blue", ls="--")
            st.pyplot(fig_h)
        with c2:
            st.metric("Retorno Projetado", f"{pred_u:.2%}")
            st.metric("Ibov Alvo", f"{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")
            rmse = np.sqrt(mean_squared_error(res_bt["Real"], res_bt["Prev"]))
            st.caption(f"RMSE Histórico: {rmse:.4f}")

# -------------------
# 3. Tabela de Sensibilidade (Stress Test)
# -------------------
st.divider()
st.header("3. Análise de Sensibilidade (Juros vs Dólar)")
st.write("Como a projeção de **1 Mês** muda se variarmos o Juros e o Dólar?")

# Criar ranges de variação (+/- 10% em 5 passos)
juros_range = np.linspace(u_inputs[0]*0.9, u_inputs[0]*1.1, 5)
dolar_range = np.linspace(u_inputs[1]*0.9, u_inputs[1]*1.1, 5)

sens_matrix = np.zeros((len(juros_range), len(dolar_range)))
mdl_sens, scaler_sens = modelos_treinados["1 Mês"]

for i, j_val in enumerate(juros_range):
    for j, d_val in enumerate(dolar_range):
        # Cria cenário: Juros e Dólar variam, PIB e Inflação fixos no input do usuário
        scenario = np.array([[j_val, d_val, u_inputs[2], u_inputs[3]]])
        sens_matrix[i, j] = mdl_sens.predict(scaler_sens.transform(scenario))[0]

df_sens = pd.DataFrame(
    sens_matrix, 
    index=[f"Selic {x:.2f}%" for x in juros_range],
    columns=[f"Dólar R${x:.2f}" for x in dolar_range]
)

# Estilização para virar um heatmap de tabela
st.dataframe(
    df_sens.style.format("{:.2%}")
    .background_gradient(cmap="RdYlGn", axis=None)
)
st.caption("Verde: Maior retorno projetado | Vermelho: Menor retorno projetado.")
