import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from bcdata import sgs

# Configuração para o pandas_datareader aceitar o yfinance
yf.pdr_override()

st.set_page_config(layout="wide", page_title="Dashboard Macro Ibovespa")

# --- Estilização ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Projeção Ex-Ante Ibovespa")
st.caption("Modelo Ridge Regression utilizando indicadores macroeconômicos defasados.")

# -------------------
# Sidebar
# -------------------
st.sidebar.header("⚙️ Configurações do Modelo")
window_type = st.sidebar.selectbox("Tipo de Janela:", ["Expanding", "Rolling"])
rolling_window_size = st.sidebar.slider("Janela Móvel (meses)", 12, 60, 36)

st.sidebar.header("🔮 Cenários para Projeção")
juros_input = st.sidebar.number_input("Selic Meta (%)", 0.0, 20.0, 10.75)
dolar_input = st.sidebar.number_input("Câmbio (BRL/USD)", 0.0, 10.0, 5.20)
pib_input = st.sidebar.number_input("Expectativa PIB (%)", -5.0, 10.0, 2.0)
inflacao_input = st.sidebar.number_input("IPCA Anualizado (%)", 0.0, 20.0, 4.5)
juros_usa_input = st.sidebar.number_input("Fed Funds Rate (%)", 0.0, 10.0, 5.25)

# -------------------
# Carregamento de Dados
# -------------------
@st.cache_data(ttl=3600)
def load_data():
    start_date = "2010-01-01"
    try:
        ibov = yf.download("^BVSP", start=start_date, interval="1mo")['Adj Close']
        dolar = yf.download("USDBRL=X", start=start_date, interval="1mo")['Adj Close']
        ipca = sgs.get({'inflacao': 433}, start=start_date)
        selic = sgs.get({'juros_brasil': 4390}, start=start_date)
        pib = sgs.get({'pib': 438}, start=start_date)
        juros_usa = pdr.get_data_fred('FEDFUNDS', start=start_date)
        
        df = pd.DataFrame(index=ibov.index)
        df['ibov'] = ibov
        df['dolar'] = dolar
        df = df.join([selic, ipca, pib, juros_usa], how='left')
        df = df.ffill().dropna()
        df['target_ret'] = df['ibov'].pct_change().shift(-1)
        return df.dropna()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.warning("Aguardando carregamento de dados ou erro nas APIs externas.")
    st.stop()

# -------------------
# Processamento e Modelo
# -------------------
features = ["juros_brasil", "dolar", "pib", "inflacao", "FEDFUNDS"]
X = df_raw[features]
y = df_raw["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treino Final para Importância e Projeção
final_model = Ridge(alpha=1.0)
final_model.fit(X_scaled, y)

# -------------------
# Dashboard Layout
# -------------------
st.divider()
c1, c2, c3 = st.columns(3)

# Cálculo da Projeção
user_data = np.array([[juros_input, dolar_input, pib_input, inflacao_input, juros_usa_input]])
user_data_scaled = scaler.transform(user_data)
proj_retorno = final_model.predict(user_data_scaled)[0]

with c1:
    st.metric("Retorno Esperado (Próx. Mês)", f"{proj_retorno:.2%}")
with c2:
    st.metric("Ibov Alvo Estimado", f"{df_raw['ibov'].iloc[-1] * (1 + proj_retorno):,.0f}")
with c3:
    st.metric("Aderência do Modelo (R²)", f"{final_model.score(X_scaled, y):.2f}")

# -------------------
# Gráficos Adicionais
# -------------------
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("🎯 Sensibilidade das Variáveis (Feature Importance)")
    # Extração dos coeficientes
    importances = pd.Series(final_model.coef_, index=features).sort_values()
    
    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    colors = ['red' if x < 0 else 'green' for x in importances]
    importances.plot(kind='barh', color=colors, ax=ax_imp)
    ax_imp.set_title("Impacto no Retorno do Ibovespa (Coeficientes Ridge)")
    ax_imp.set_xlabel("Peso no Modelo (Normalizado)")
    ax_imp.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig_imp)
    st.info("💡 **Interpretação:** Barras à direita indicam correlação positiva com o retorno futuro; à esquerda, correlação negativa.")

with col_graph2:
    st.subheader("📊 Distribuição dos Resíduos")
    y_pred_all = final_model.predict(X_scaled)
    residuals = y - y_pred_all
    
    fig_res, ax_res = plt.subplots(figsize=(8, 6))
    ax_res.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax_res.axvline(0, color='red', linestyle='--')
    ax_res.set_title("Erro de Previsão (Resíduos)")
    st.pyplot(fig_res)

# -------------------
# Histórico de Backtest (Abaixo)
# -------------------
st.divider()
st.subheader("🔄 Histórico de Backtest vs Real")

def run_backtest(X_in, y_in, window, size):
    preds = []
    start = 48 
    for i in range(start, len(y_in)):
        X_train = X_in[:i] if window == "Expanding" else X_in[max(0, i-size):i]
        y_train = y_in[:i] if window == "Expanding" else y_in[max(0, i-size):i]
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        preds.append(model.predict(X_in[i:i+1])[0])
    return pd.Series(preds, index=y_in.index[start:])

y_pred_series = run_backtest(X_scaled, y, window_type, rolling_window_size)
y_true_series = y.loc[y_pred_series.index]

fig_final, ax_final = plt.subplots(figsize=(12, 4))
ax_final.plot(y_true_series.cumsum(), label="Real (Acumulado)", color="#1f77b4", lw=2)
ax_final.plot(y_pred_series.cumsum(), label="Modelo (Acumulado)", color="#ff7f0e", linestyle="--")
ax_final.set_title("Performance Acumulada no Período de Teste")
ax_final.legend()
st.pyplot(fig_final)
