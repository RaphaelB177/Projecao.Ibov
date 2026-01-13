import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from bcb import sgs

# Configuração para o pandas_datareader
yf.pdr_override()

st.set_page_config(layout="wide", page_title="Dashboard Macro Ibovespa")

st.title("📈 Projeção Ex-Ante Ibovespa")
st.caption("Modelo Ridge Regression utilizando indicadores macroeconômicos via SGS/BCB e Yahoo Finance.")

# -------------------
# Sidebar
# -------------------
st.sidebar.header("⚙️ Configurações")
window_type = st.sidebar.selectbox("Tipo de Janela:", ["Expanding", "Rolling"])
rolling_window_size = st.sidebar.slider("Janela Móvel (meses)", 12, 60, 36)

st.sidebar.header("🔮 Cenários")
juros_input = st.sidebar.number_input("Selic Meta (%)", 0.0, 20.0, 10.75)
dolar_input = st.sidebar.number_input("Câmbio (BRL/USD)", 0.0, 10.0, 5.20)
pib_input = st.sidebar.number_input("Expectativa PIB (%)", -5.0, 10.0, 2.0)
inflacao_input = st.sidebar.number_input("IPCA Anualizado (%)", 0.0, 20.0, 4.5)
juros_usa_input = st.sidebar.number_input("Fed Funds Rate (%)", 0.0, 10.0, 5.25)

# -------------------
# Carregamento de Dados (Ajustado para bcb)
# -------------------
@st.cache_data(ttl=3600)
def load_data():
    start_date = "2010-01-01"
    try:
        # 1. Ibov e Dólar - Pegamos diário e resamplamos para garantir consistência
        ibov = yf.download("^BVSP", start=start_date)['Adj Close'].resample('ME').last()
        dolar = yf.download("USDBRL=X", start=start_date)['Adj Close'].resample('ME').last()
        
        # 2. Dados Banco Central (SGS) via biblioteca bcb
        # 433: IPCA, 4390: SELIC, 438: PIB
        ipca = sgs.get({'inflacao': 433}, start=start_date)
        selic = sgs.get({'juros_brasil': 4390}, start=start_date)
        pib = sgs.get({'pib': 438}, start=start_date)
        
        # 3. Juros USA (FRED)
        juros_usa = pdr.get_data_fred('FEDFUNDS', start=start_date)
        
        # Consolidação
        df = pd.DataFrame(index=ibov.index)
        df['ibov'] = ibov
        df['dolar'] = dolar
        
        # Join com tratamento de frequência
        for d in [selic, ipca, pib, juros_usa]:
            d.index = pd.to_datetime(d.index)
            df = df.join(d.resample('ME').last(), how='left')
        
        df = df.ffill().dropna()
        df['target_ret'] = df['ibov'].pct_change().shift(-1)
        return df.dropna()
    except Exception as e:
        st.error(f"Erro na extração de dados: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.info("Aguardando sincronização com as APIs (BCB, Yahoo, FRED)...")
    st.stop()

# -------------------
# Modelo e UI
# -------------------
features = ["juros_brasil", "dolar", "pib", "inflacao", "FEDFUNDS"]
X = df_raw[features]
y = df_raw["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = Ridge(alpha=1.0).fit(X_scaled, y)

# Layout de métricas
st.divider()
c1, c2, c3 = st.columns(3)
user_data_scaled = scaler.transform([[juros_input, dolar_input, pib_input, inflacao_input, juros_usa_input]])
proj_retorno = final_model.predict(user_data_scaled)[0]

with c1:
    st.metric("Retorno Esperado (M+1)", f"{proj_retorno:.2%}")
with c2:
    st.metric("Ibov Alvo", f"{df_raw['ibov'].iloc[-1] * (1 + proj_retorno):,.0f}")
with c3:
    st.metric("R² do Modelo", f"{final_model.score(X_scaled, y):.2f}")

# Gráfico de Importância
st.subheader("🎯 O que está movendo a projeção?")
importances = pd.Series(final_model.coef_, index=features).sort_values()
fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
importances.plot(kind='barh', color=['red' if x < 0 else 'green' for x in importances], ax=ax_imp)
st.pyplot(fig_imp)

# Gráfico de Backtest
st.subheader("🔄 Histórico: Real vs Modelo")
def run_bt(X_s, y_s, w, s):
    preds = []
    idx = range(48, len(y_s))
    for i in idx:
        xt = X_s[:i] if w == "Expanding" else X_s[max(0, i-s):i]
        yt = y_s[:i] if w == "Expanding" else y_s[max(0, i-s):i]
        preds.append(Ridge(alpha=1.0).fit(xt, yt).predict(X_s[i:i+1])[0])
    return pd.Series(preds, index=y_s.index[48:])

y_p = run_bt(X_scaled, y, window_type, rolling_window_size)
fig_bt, ax_bt = plt.subplots(figsize=(12, 4))
ax_bt.plot(y.loc[y_p.index].cumsum(), label="Real (Acumulado)")
ax_bt.plot(y_p.cumsum(), label="Modelo (Acumulado)", ls="--")
ax_bt.legend()
st.pyplot(fig_bt)
