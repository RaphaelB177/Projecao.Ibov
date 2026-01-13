import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------
# Configurações do App
# -------------------
st.set_page_config(layout="wide")
st.title("Projeção Ex-Ante do Ibovespa com Dados Macro Públicos")

# -------------------
# Sidebar Inputs
# -------------------
st.sidebar.header("Parâmetros Macro Econômicos (ex-ante)")
window_type = st.sidebar.selectbox("Tipo de Janela para Backtest:", ["Expanding", "Rolling"])
rolling_window_size = st.sidebar.slider("Tamanho da Rolling Window (meses)", 12, 60, 36)

# Inputs do usuário para projeção atual
st.sidebar.header("Parâmetros de Entrada do Usuário")
juros_input = st.sidebar.number_input("Taxa de Juros Brasil (%)", 0.0, 20.0, 5.0, 0.1)
dolar_input = st.sidebar.number_input("Cotação do Dólar (R$)", 0.0, 10.0, 5.0, 0.01)
pib_input = st.sidebar.number_input("Variação do PIB (%)", -10.0, 10.0, 2.0, 0.1)
inflacao_input = st.sidebar.number_input("Inflação (%)", 0.0, 20.0, 4.0, 0.1)
juros_americano_input = st.sidebar.number_input("Juros Americano (%)", 0.0, 20.0, 3.0, 0.1)

# -------------------
# Função para baixar dados
# -------------------
@st.cache_data(show_spinner=True)
def load_data():
    # Ibovespa e Dólar
    ibov = yf.download("^BVSP", start="2000-01-01", progress=False)
    dolar = yf.download("USDBRL=X", start="2000-01-01", progress=False)
    df = pd.DataFrame({
        "ibov": ibov["Adj Close"],
        "dolar": dolar["Adj Close"]
    }).dropna()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').last()

    # Juros americano via FRED
    try:
        juros_usa = pdr.DataReader("FEDFUNDS", "fred", df.index.min(), df.index.max())
    except:
        juros_usa = pd.DataFrame(3.0, index=df.index, columns=["FEDFUNDS"])
    juros_usa = juros_usa.resample('M').ffill()

    # Dados brasileiros via BCData (IPCA, SELIC, PIB)
    # IPCA: código 433 (mensal)
    try:
        from bcdata import sgs
        ipca = sgs.get({'IPCA': 433}, start=df.index.min(), end=df.index.max())
        ipca.index = pd.to_datetime(ipca.index)
        ipca = ipca.resample('M').ffill()
    except:
        ipca = pd.DataFrame(4.0, index=df.index, columns=["IPCA"])

    # SELIC: código 4189 (mensal)
    try:
        selic = sgs.get({'SELIC': 4189}, start=df.index.min(), end=df.index.max())
        selic.index = pd.to_datetime(selic.index)
        selic = selic.resample('M').ffill()
    except:
        selic = pd.DataFrame(5.0, index=df.index, columns=["SELIC"])

    # PIB trimestral (ex: 438)
    try:
        pib = sgs.get({'PIB': 438}, start=df.index.min(), end=df.index.max())
        pib.index = pd.to_datetime(pib.index)
        pib = pib.resample('M').ffill()  # converter para mensal
    except:
        pib = pd.DataFrame(2.0, index=df.index, columns=["PIB"])

    # Juntar tudo
    df = df.join([selic.rename(columns={'SELIC':'juros_brasil'}), 
                  dolar.rename(columns={'Adj Close':'dolar'}), 
                  pib.rename(columns={'PIB':'pib'}), 
                  ipca.rename(columns={'IPCA':'inflacao'}), 
                  juros_usa.rename(columns={'FEDFUNDS':'juros_americano'})], how='left')
    df = df.dropna()
    df['ibov_ret'] = df['ibov'].pct_change()
    df = df.dropna()
    return df

df = load_data()

# -------------------
# Preparar X e y
# -------------------
X = df[["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]]
y = df["ibov_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------
# Função de Backtest
# -------------------
def backtest_ridge(X, y, horizon_months, alpha, window_type="Expanding", rolling_size=36):
    preds = []
    test_idx = []

    if window_type=="Expanding":
        start_idx = 60
        for i in range(start_idx, len(y)-horizon_months):
            X_train = X[:i]
            y_train = y[:i]
            X_test = X[i:i+horizon_months]
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds.extend(y_pred)
            test_idx.extend(y.index[i:i+horizon_months])
    else: # Rolling
        for i in range(rolling_size, len(y)-horizon_months):
            X_train = X[i-rolling_size:i]
            y_train = y[i-rolling_size:i]
            X_test = X[i:i+horizon_months]
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds.extend(y_pred)
            test_idx.extend(y.index[i:i+horizon_months])

    y_true = y.loc[test_idx]
    y_pred_series = pd.Series(preds, index=test_idx)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_series))
    mae = mean_absolute_error(y_true, y_pred_series)

    return y_pred_series, y_true, rmse, mae

# -------------------
# Backtest 1,6,12 meses
# -------------------
alphas = {"1 mês":0.5, "6 meses":0.5, "12 meses":1.0}
horizons = {"1 mês":1, "6 meses":6, "12 meses":12}

st.header("Backtest Ex-Ante do Modelo Ridge")
results = {}

for label, horizon in horizons.items():
    y_pred, y_true, rmse, mae = backtest_ridge(X_scaled, y, horizon, alphas[label],
                                               window_type=window_type,
                                               rolling_size=rolling_window_size)
    results[label] = {"y_pred":y_pred, "y_true":y_true, "rmse":rmse, "mae":mae}
    st.subheader(f"Horizonte: {label}")
    st.write(f"RMSE: {rmse:.5f} | MAE: {mae:.5f}")

    resid = y_true - y_pred
    std_resid = resid.std()
    lower = y_pred - 1.96*std_resid
    upper = y_pred + 1.96*std_resid

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(y_true.index, y_true, label="Real", color="black")
    ax.plot(y_pred.index, y_pred, label="Previsto", color="blue")
    ax.fill_between(y_pred.index, lower, upper, color='blue', alpha=0.2, label="Intervalo 95%")
    ax.set_title(f"Backtest Ridge - Horizonte {label}")
    ax.legend()
    st.pyplot(fig)

# -------------------
# Projeção atual com input do usuário
# -------------------
st.header("Projeção Atual com Inputs do Usuário")

X_user = np.array([[juros_input, dolar_input, pib_input, inflacao_input, juros_americano_input]])
X_user_scaled = scaler.transform(X_user)

model_final = Ridge(alpha=alphas["1 mês"])
model_final.fit(X_scaled, y)

pred_user = model_final.predict(X_user_scaled)[0]
ultimo_preco = df['ibov'].iloc[-1]
preco_proj = ultimo_preco*(1+pred_user)

st.write(f"Projeção retorno mensal Ibovespa (1 mês): **{pred_user*100:.3f}%**")
st.write(f"Preço estimado Ibovespa em 1 mês: **{preco_proj:.2f}**")

resid_all = y - model_final.predict(X_scaled)
std_resid_all = resid_all.std()
ic_lower = pred_user - 1.96*std_resid_all
ic_upper = pred_user + 1.96*std_resid_all

st.write(f"Intervalo de confiança 95% retorno mensal: [{ic_lower*100:.3f}%, {ic_upper*100:.3f}%]")
