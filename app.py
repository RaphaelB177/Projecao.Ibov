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
    
    # 1. Ibov (Tratamento Multi-index robusto)
    try:
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
    except:
        st.error("Erro ao carregar dados do Yahoo Finance.")
        st.stop()

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

    # Join Macro (Somente se houver dados)
    for cod, nome in [(1, 'dolar'), (433, 'inflacao'), (4390, 'juros_brasil'), (438, 'pib')]:
        m_data = get_sgs_safe(cod, nome)
        if not m_data.empty: 
            df = df.join(m_data, how='left')
    
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

# Sidebar - Detecção Automática de Features Presentes
features_all = ["juros_brasil", "dolar", "pib", "inflacao"]
features_presentes = [f for f in features_all if f in df_full.columns]

if not features_presentes:
    st.error("Nenhum dado macroeconômico foi carregado. Verifique a conexão com o Banco Central.")
    st.stop()

st.sidebar.header("Cenário e Modelo")
window_type = st.sidebar.selectbox("Janela Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Janela Móvel", 12, 60, 36)

u_inputs = []
for f in features_presentes:
    # Verificação de segurança para evitar o KeyError
    ultimo_valor = float(df_full[f].iloc[-1])
    val = st.sidebar.number_input(f"Expectativa {f}", value=ultimo_valor, step=0.01)
    u_inputs.append(val)

# -------------------
# 1. Mapa de Calor (Heatmap)
# -------------------

st.header("1. Força dos Indicadores (Correlação)")
corr_cols = features_presentes + ['ret_1m', 'ret_6m', 'ret_12m']
corr_matrix = df_full[corr_cols].corr()
corr_target = corr_matrix.loc[features_presentes, ['ret_1m', 'ret_6m', 'ret_12m']]

fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
sns.heatmap(corr_target.T, annot=True, cmap="RdYlGn", center=0, ax=ax_corr)
ax_corr.set_title("Correlação: Indicador Hoje vs Retorno Futuro")
st.pyplot(fig_corr)

# -------------------
# 2. Projeções e Backtest
# -------------------
st.header("2. Projeções por Horizonte")
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_treinados = {}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, col_target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[col_target])
        if len(df_h) < 48:
            st.warning(f"Histórico insuficiente para {label}.")
            continue
            
        X, y = df_h[features_presentes], df_h[col_target]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Backtest
        start_idx = 48
        preds_bt, actuals_bt = [], []
        for j in range(start_idx, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            mdl = Ridge(alpha=0.5).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        res_bt = pd.DataFrame({"Real": actuals_bt, "Prev": preds_bt}, index=y.index[start_idx:])
        
        # Modelo Final para Projeção
        mdl_f = Ridge(alpha=0.5).fit(X_s, y)
        modelos_treinados[label] = (mdl_f, scaler)
        pred_u = mdl_f.predict(scaler.transform([u_inputs]))[0]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_h, ax_h = plt.subplots(figsize=(10, 4))
            ax_h.plot(res_bt["Real"].cumsum(), label="Real", color="black", alpha=0.7)
            ax_h.plot(res_bt["Prev"].cumsum(), label="Previsto", color="#1f77b4", ls="--")
            ax_h.set_title(f"Performance Acumulada - {label}")
            ax_h.legend()
            st.pyplot(fig_h)
        with c2:
            st.metric("Retorno Projetado", f"{pred_u:.2%}")
            st.metric("Ibov Alvo", f"{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")
            rmse = np.sqrt(mean_squared_error(res_bt["Real"], res_bt["Prev"]))
            st.caption(f"RMSE Histórico: {rmse:.4f}")

# -------------------
# 3. Tabela de Sensibilidade
# -------------------

st.divider()
if "juros_brasil" in features_presentes and "dolar" in features_presentes:
    st.header("3. Análise de Sensibilidade (Stress Test)")
    st.write("Matriz de Retorno Projetado (1 Mês) variando Juros e Dólar:")

    j_idx = features_presentes.index("juros_brasil")
    d_idx = features_presentes.index("dolar")

    # Gerar ranges (+/- 10%)
    j_range = np.linspace(u_inputs[j_idx]*0.9, u_inputs[j_idx]*1.1, 5)
    d_range = np.linspace(u_inputs[d_idx]*0.9, u_inputs[d_idx]*1.1, 5)

    sens_matrix = np.zeros((5, 5))
    mdl_s, scaler_s = modelos_treinados["1 Mês"]

    for i, j_val in enumerate(j_range):
        for j, d_val in enumerate(d_range):
            scenario = list(u_inputs)
            scenario[j_idx] = j_val
            scenario[d_idx] = d_val
            sens_matrix[i, j] = mdl_s.predict(scaler_s.transform([scenario]))[0]

    df_sens = pd.DataFrame(
        sens_matrix, 
        index=[f"Selic {x:.2f}%" for x in j_range],
        columns=[f"Dólar R${x:.2f}" for x in d_range]
    )

    st.dataframe(df_sens.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=None))
