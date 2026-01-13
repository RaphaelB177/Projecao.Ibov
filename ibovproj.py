import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Projeção Ibovespa")

# ------------------------------
# Funções auxiliares
# ------------------------------

# Métricas
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Backtest Ridge (expanding ou rolling)
def backtest_ridge(df, features, h, alpha=1.0, window_type="Expanding", window_size=120):
    results = []
    for t in range(window_size, len(df)-h):
        if window_type=="Expanding":
            train = df.iloc[:t]
        else:
            train = df.iloc[t-window_size:t]

        y_train = train['r_ibov'].iloc[:-h]
        X_train = train[features].iloc[:-h]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y_train)

        X_t = scaler.transform(df[features].iloc[t].values.reshape(1,-1))
        forecast_return = model.predict(X_t)[0]

        ibov_t = df['ibov'].iloc[t]
        ibov_real = df['ibov'].iloc[t+h]
        results.append({
            'date': df.index[t],
            'forecast_return': forecast_return,
            'real_return': np.log(ibov_real) - np.log(ibov_t),
            'forecast_ibov': ibov_t * np.exp(forecast_return)
        })
    return pd.DataFrame(results)

# Random Walk
def random_walk_forecast(df, h, start_window=120):
    results = []
    for t in range(start_window, len(df)-h):
        ibov_t = df.iloc[t]['ibov']
        ibov_real = df.iloc[t+h]['ibov']
        results.append({
            'date': df.index[t],
            'forecast_return': 0.0,
            'real_return': np.log(ibov_real) - np.log(ibov_t),
            'forecast_ibov': ibov_t
        })
    return pd.DataFrame(results)

# AR(1)
def ar1_forecast(df, h, start_window=120):
    results = []
    for t in range(start_window, len(df)-h):
        train = df['r_ibov'].iloc[:t].dropna()
        model = sm.tsa.ARIMA(train, order=(1,0,0)).fit()
        r_forecast = model.forecast()[0]
        forecast_return = h * r_forecast

        ibov_t = df['ibov'].iloc[t]
        ibov_real = df['ibov'].iloc[t+h]
        results.append({
            'date': df.index[t],
            'forecast_return': forecast_return,
            'real_return': np.log(ibov_real) - np.log(ibov_t),
            'forecast_ibov': ibov_t * np.exp(forecast_return)
        })
    return pd.DataFrame(results)

# Subperiodos
SUBPERIODS = {
    "Crisis_2008": ("2008-09-01", "2009-06-30"),
    "Covid": ("2020-03-01", "2021-06-30"),
    "Election_2002": ("2002-04-01", "2003-03-31"),
    "Election_2006": ("2006-04-01", "2007-03-31"),
    "Election_2010": ("2010-04-01", "2011-03-31"),
    "Election_2014": ("2014-04-01", "2015-03-31"),
    "Election_2018": ("2018-04-01", "2019-03-31"),
    "Election_2022": ("2022-04-01", "2023-03-31"),
}

def evaluate_subperiods(results, subperiods):
    evals = []
    for name, (start, end) in subperiods.items():
        mask = (results['date'] >= start) & (results['date'] <= end)
        sub = results.loc[mask]
        if len(sub)<5: 
            continue
        evals.append({
            'period': name,
            'observations': len(sub),
            'RMSE': rmse(sub['real_return'], sub['forecast_return']),
            'MAE': mae(sub['real_return'], sub['forecast_return']),
            'Directional_Accuracy': (np.sign(sub['real_return'])==np.sign(sub['forecast_return'])).mean()
        })
    return pd.DataFrame(evals)

# Adiciona banda de confiança
def add_confidence_band(results, confidence=0.95):
    z = {0.68:1,0.9:1.645,0.95:1.96}[confidence]
    sigma = np.std(results['forecast_return'] - results['real_return'])
    results['upper'] = results['forecast_ibov'] * np.exp(z*sigma)
    results['lower'] = results['forecast_ibov'] * np.exp(-z*sigma)
    return results

# ------------------------------
# Carregar dados
# ------------------------------

@st.cache_data
def load_base_results():
    return pd.read_parquet("data/processed_base_results.parquet")

@st.cache_data
def load_macro_data():
    return pd.read_parquet("data/macro_data.parquet")

base_results = load_base_results()
macro_data = load_macro_data()
ALL_FEATURES = [c for c in macro_data.columns if c not in ['ibov','r_ibov']]

# ------------------------------
# Layout Streamlit
# ------------------------------

st.title("📈 Projeção do Ibovespa — Estudo Econométrico")

tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Estudo Base",
    "⚠️ Crises & Eleições",
    "🧪 Projeção do Usuário",
    "📊 Comparação"
])

# ------------------------------
# Tab 1 — Estudo Base
# ------------------------------
with tab1:
    st.subheader("Backtest Ex Ante — Modelo Base")
    h_base = st.selectbox("Horizonte (meses) - Base", [1,6,12], key="base_h")
    data_base = base_results[base_results["horizon"]==h_base]
    st.line_chart(data_base.set_index("date")[["real_ibov","forecast_ibov"]], height=400)

# ------------------------------
# Tab 2 — Crises & Eleições
# ------------------------------
with tab2:
    st.subheader("Performance em Subperíodos")
    period = st.selectbox("Selecionar período", list(SUBPERIODS.keys()))
    start,end = SUBPERIODS[period]
    sub = data_base[(data_base["date"]>=start)&(data_base["date"]<=end)]
    st.metric("RMSE", f"{rmse(sub.real_return, sub.forecast_return):.4f}")
    st.metric("MAE", f"{mae(sub.real_return, sub.forecast_return):.4f}")
    st.line_chart((sub.forecast_return - sub.real_return).cumsum(), height=300)

# ------------------------------
# Tab 3 — Projeção do Usuário
# ------------------------------
with st.sidebar:
    st.header("🧪 Projeção do Usuário")
    user_h = st.selectbox("Horizonte", [1,6,12], key="user_h")
    alpha = st.slider("α (Ridge)", 0.01, 50.0, 1.0)
    window_type = st.radio("Janela", ["Expanding","Rolling"])
    window_size = st.slider("Rolling window (meses)", 60,180,120)
    selected_features = st.multiselect("Variáveis", ALL_FEATURES, default=ALL_FEATURES)
    confidence = st.selectbox("Bandas de Confiança", [0.68,0.9,0.95], index=2)
    run_user = st.button("Gerar Projeção do Usuário")

with tab3:
    if run_user:
        macro_df = macro_data.copy()
        results_user = backtest_ridge(
            df=macro_df,
            features=selected_features,
            h=user_h,
            alpha=alpha,
            window_type=window_type,
            window_size=window_size
        )
        results_user = add_confidence_band(results_user, confidence=confidence)
        st.subheader("Resultado — Projeção do Usuário")
        st.line_chart(results_user.set_index("date")[["forecast_ibov","upper","lower","real_ibov"]], height=400)

# ------------------------------
# Tab 4 — Comparação
# ------------------------------
with tab4:
    st.subheader("Base vs Projeção do Usuário")
    if run_user:
        compare = data_base.merge(
            results_user,
            on="date",
            suffixes=("_base","_user")
        )
        st.line_chart(compare.set_index("date")[["forecast_ibov_base","forecast_ibov_user","real_ibov_base"]], height=400)
