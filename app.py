import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import requests

# -------------------
# Configurações e Cache
# -------------------
st.set_page_config(layout="wide", page_title="Ibov Projeção Estatística")

if st.sidebar.button("🔄 Atualizar Base de Dados"):
    st.cache_data.clear()
    st.rerun()

# Funções de extração resilientes (estratégia atual)
def get_sgs_csv(codigo, nome_coluna):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',')
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df = df.rename(columns={'valor': nome_coluna}).set_index('data')
            return df[[nome_coluna]]
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=None)
def load_data_full():
    hoje = datetime.now() - timedelta(days=2)
    start_str = (hoje - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # Ibov Real
    ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
    ibov = ibov_raw['Adj Close'].iloc[:,0] if isinstance(ibov_raw.columns, pd.MultiIndex) else ibov_raw['Adj Close']
    df = ibov.resample('ME').last().to_frame('ibov')

    # Dados Macro
    dolar = get_sgs_csv(1, 'dolar')
    ipca = get_sgs_csv(433, 'inflacao')
    selic = get_sgs_csv(4390, 'juros_brasil')
    pib = get_sgs_csv(438, 'pib')
    
    for d in [dolar, ipca, selic, pib]:
        if not d.empty:
            df = df.join(d.resample('ME').last(), how='left')
    
    df = df.ffill().dropna()
    df['target_ret'] = df['ibov'].pct_change().shift(-1) # Retorno Ex-Ante
    return df.dropna()

df = load_data_full()

# -------------------
# Lógica Estatística (Recuperada do Original)
# -------------------
st.title("📈 Projeção Estatística Ibovespa")

# Sidebar - Parâmetros de Backtest
st.sidebar.header("Parâmetros Estatísticos")
window_type = st.sidebar.selectbox("Tipo de Janela:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Janela Móvel (meses)", 12, 60, 36)

features = ["juros_brasil", "dolar", "pib", "inflacao"] # Ajustado para o que o BCB entrega estável
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def backtest_ridge_full(X, y, horizon, window_type, size):
    preds, actuals, t_idx = [], [], []
    start_idx = 48 # Mínimo de 4 anos
    
    for i in range(start_idx, len(y) - horizon + 1):
        X_train = X[:i] if window_type == "Expanding" else X[i-size:i]
        y_train = y[:i] if window_type == "Expanding" else y[i-size:i]
        
        model = Ridge(alpha=0.5).fit(X_train, y_train)
        y_pred = model.predict(X[i : i+horizon])
        
        preds.extend(y_pred)
        actuals.extend(y[i : i+horizon])
        t_idx.extend(y.index[i : i+horizon])
        
    res = pd.DataFrame({"Real": actuals, "Previsto": preds}, index=t_idx)
    rmse = np.sqrt(mean_squared_error(res["Real"], res["Previsto"]))
    return res, rmse

# Execução do Backtest
st.header("1. Avaliação de Performance (Backtest)")
res_bt, rmse_val = backtest_ridge_full(X_scaled, y, 1, window_type, rolling_size)

col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 4))
    std_err = (res_bt["Real"] - res_bt["Previsto"]).std()
    ax.plot(res_bt["Real"].cumsum(), label="Real Acumulado", color="black")
    ax.plot(res_bt["Previsto"].cumsum(), label="Previsto Acumulado", color="blue", linestyle="--")
    ax.fill_between(res_bt.index, (res_bt["Previsto"] - 1.96*std_err).cumsum(), 
                    (res_bt["Previsto"] + 1.96*std_err).cumsum(), color='blue', alpha=0.1, label="IC 95%")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.metric("RMSE do Modelo", f"{rmse_val:.4f}")
    st.write("O erro médio quadrático indica a volatilidade da falha do modelo no passado.")

# -------------------
# Projeção Atual (Com Inputs do Usuário)
# -------------------
st.divider()
st.header("2. Projeção para o Próximo Mês")

st.sidebar.header("Inputs para Projeção")
user_vals = []
for f in features:
    v = st.sidebar.number_input(f"Valor esperado: {f}", value=float(df[f].iloc[-1]))
    user_vals.append(v)

model_final = Ridge(alpha=0.5).fit(X_scaled, y)
pred_user = model_final.predict(scaler.transform([user_vals]))[0]

c1, c2 = st.columns(2)
c1.metric("Retorno Projetado", f"{pred_ret_user:.2%}")
c2.metric("Preço Estimado", f"{df['ibov'].iloc[-1]*(1+pred_user):,.0f}")

st.write(f"**Intervalo de Incerteza (95%):** [{(pred_user - 1.96*std_err)*100:.2f}% a {(pred_user + 1.96*std_err)*100:.2f}%]")
