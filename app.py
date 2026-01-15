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
# 1. Configurações Iniciais
# -------------------
st.set_page_config(layout="wide", page_title="Master Ibov Dashboard", page_icon="📈")

@st.cache_data(ttl=None)
def load_data_master():
    hoje = datetime.now() - timedelta(days=2)
    start_str = (hoje - timedelta(days=365*12)).strftime('%Y-%m-%d')
    
    # Download Ibov
    try:
        ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
        col = 'Adj Close' if 'Adj Close' in ibov_raw.columns else 'Close'
        df = ibov_raw[col].resample('ME').last().to_frame('ibov')
    except:
        st.error("Erro Yahoo Finance. Usando dados base 2026.")
        df = pd.DataFrame({'ibov': [161973]}, index=[pd.to_datetime('2026-01-14')])

    # Fallback para caso o BCB falhe (Dados de Jan/2026)
    fallbacks = {'dolar': 5.37, 'inflacao': 4.4, 'juros_brasil': 15.0, 'pib': 3.0}

    def get_sgs(codigo, nome):
        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
            r = requests.get(url, timeout=10)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            return d.rename(columns={'valor': nome}).set_index('data').resample('ME').last()
        except:
            return pd.DataFrame({nome: [fallbacks[nome]] * len(df)}, index=df.index)

    # Join das Séries
    for cod, nome in [(1, 'dolar'), (433, 'inflacao'), (4390, 'juros_brasil'), (438, 'pib')]:
        df = df.join(get_sgs(cod, nome), how='left')
    
    df = df.ffill().bfill()
    
    # Horizontes de Retorno (Lógica Original)
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    return df

df_full = load_data_master()

# -------------------
# 2. Interface e Inputs
# -------------------
st.title("📊 Master Dashboard: Projeção Ibovespa")

st.sidebar.header("Parâmetros do Modelo")
window_type = st.sidebar.selectbox("Tipo de Janela Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Tamanho Rolling (meses):", 12, 60, 36)

st.sidebar.header("Inputs de Projeção")
features = ['juros_brasil', 'dolar', 'inflacao', 'pib']
u_inputs = [st.sidebar.number_input(f"Valor {f}", value=float(df_full[f].iloc[-1])) for f in features]

# -------------------
# 3. Análise de Correlação (Heatmap)
# -------------------
st.header("1. Correlação Macro vs Retornos")
corr_target = df_full[features + ['ret_1m', 'ret_6m', 'ret_12m']].corr().loc[features, ['ret_1m', 'ret_6m', 'ret_12m']]
fig_corr, ax_corr = plt.subplots(figsize=(10, 3))
sns.heatmap(corr_target.T, annot=True, cmap="RdYlGn", center=0, ax=ax_corr)
st.pyplot(fig_corr)

# -------------------
# 4. Horizontes, Backtest e Projeção
# -------------------
st.header("2. Projeções e Performance")
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_finais = {}

tabs = st.tabs(list(horizontes.keys()))
for i, (label, target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[target])
        X, y = df_h[features], df_h[target]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Backtest (Lógica Original)
        preds_bt, actuals_bt = [], []
        start = 48
        for j in range(start, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            mdl = Ridge(alpha=0.5).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        # Gráfico e Projeção
        res_bt = pd.Series(preds_bt, index=y.index[start:])
        mdl_f = Ridge(alpha=0.5).fit(X_s, y)
        modelos_finais[label] = (mdl_f, scaler)
        pred_val = mdl_f.predict(scaler.transform([u_inputs]))[0]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y[start:].cumsum(), label="Real", color="black")
            ax.plot(res_bt.cumsum(), label="Modelo", color="blue", ls="--")
            ax.set_title(f"Backtest Acumulado - {label}")
            ax.legend()
            st.pyplot(fig)
        with c2:
            st.metric("Retorno Projetado", f"{pred_val:.2%}")
            st.metric("Alvo Estimado", f"{df_full['ibov'].iloc[-1]*(1+pred_val):,.0f}")
            st.write(f"Incerteza (RMSE): {np.sqrt(mean_squared_error(y[start:], res_bt)):.4f}")

# -------------------
# 5. Tabela de Sensibilidade
# -------------------

st.divider()
st.header("3. Stress Test (Juros vs Dólar)")
j_range = np.linspace(u_inputs[0]*0.9, u_inputs[0]*1.1, 7)
d_range = np.linspace(u_inputs[1]*0.9, u_inputs[1]*1.1, 7)
mdl_s, scaler_s = modelos_finais["1 Mês"]

matrix = [[mdl_s.predict(scaler_s.transform([[j, d, u_inputs[2], u_inputs[3]]]))[0] for d in d_range] for j in j_range]
df_sens = pd.DataFrame(matrix, index=[f"Selic {x:.2f}%" for x in j_range], columns=[f"Dólar R${x:.2f}" for x in d_range])
st.dataframe(df_sens.style.format("{:.2%Short}").background_gradient(cmap="RdYlGn", axis=None))

# -------------------
# 6. Base de Dados Bruta
# -------------------
st.divider()
st.header("4. Base de Dados Bruta")
st.dataframe(df_full.tail(15))
