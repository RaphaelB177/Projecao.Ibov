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
# Configurações e Coleta de Dados
# -------------------
st.set_page_config(layout="wide", page_title="Master Ibov Dashboard", page_icon="📈")

@st.cache_data(ttl=3600) # Atualiza a cada hora
def load_data_master():
    hoje = datetime.now() - timedelta(days=1)
    start_str = (hoje - timedelta(days=365*15)).strftime('%Y-%m-%d')
    
    # 1. Ibov e Dólar (Yahoo Finance - Tickers BRL=X e ^BVSP)
    try:
        # Baixamos Ibov e Dólar juntos para garantir sincronia de datas
        data_yf = yf.download(["^BVSP", "BRL=X"], start=start_str, progress=False)
        
        # Tratamento de MultiIndex
        if isinstance(data_yf.columns, pd.MultiIndex):
            ibov = data_yf['Adj Close']['^BVSP']
            dolar = data_yf['Adj Close']['BRL=X']
        else:
            ibov = data_yf['^BVSP']
            dolar = data_yf['BRL=X']
            
        df = pd.DataFrame({'ibov': ibov, 'dolar': dolar})
        df = df.resample('ME').last()
    except:
        # Fallback de emergência caso Yahoo falhe totalmente
        dates = pd.date_range(end=hoje, periods=100, freq='ME')
        df = pd.DataFrame({'ibov': np.nan, 'dolar': np.nan}, index=dates)

    # 2. Séries do Banco Central (SGS)
    def get_sgs(codigo, nome):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
        try:
            r = requests.get(url, timeout=10)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            return d.rename(columns={'valor': nome}).set_index('data').resample('ME').last()
        except: return pd.DataFrame()

    # Selic Meta (432), IPCA (433), PIB (438)
    for cod, nome in [(433, 'inflacao'), (432, 'juros_brasil'), (438, 'pib')]:
        sgs_df = get_sgs(cod, nome)
        if not sgs_df.empty:
            df = df.join(sgs_df, how='left')

    # Preenchimento de lacunas com ffill (dados macro demoram mais a sair que o dólar)
    df = df.ffill().dropna(subset=['ibov', 'dolar'])
    
    # 3. AJUSTE ESTATÍSTICO: Acumulado 12 Meses para Inflação e PIB
    df['inflacao'] = df['inflacao'].rolling(12).sum() 
    df['pib'] = df['pib'].rolling(12).mean() 
    
    # Alvos de Retorno
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    
    return df.dropna(subset=['inflacao', 'pib'])

df_full = load_data_master()

# --- Interface ---
st.title("📊 Master Dashboard: Projeção Ibovespa")
st.caption(f"Dados atualizados até: {df_full.index[-1].strftime('%d/%m/%Y')}")

# Sidebar
st.sidebar.title("🛠️ Configurações")
window_type = st.sidebar.selectbox("Tipo de Janela Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Tamanho Rolling (meses):", 12, 60, 36)

st.sidebar.divider()
st.sidebar.header("🔮 Cenário Futuro")
st.sidebar.info("Insira as expectativas para os próximos 12 meses.")

# Inputs com os últimos valores REAIS da base
juros_in = st.sidebar.number_input("Selic Meta (%)", value=float(df_full['juros_brasil'].iloc[-1]), format="%.2f")
dolar_in = st.sidebar.number_input("Dólar Atual/Esperado (R$)", value=float(df_full['dolar'].iloc[-1]), format="%.2f")
infla_in = st.sidebar.number_input("Inflação Acumulada 12M (%)", value=float(df_full['inflacao'].iloc[-1]), format="%.2f")
pib_in = st.sidebar.number_input("Crescimento PIB 12M (%)", value=float(df_full['pib'].iloc[-1]), format="%.2f")

u_inputs = [juros_in, dolar_in, infla_in, pib_in]
features = ['juros_brasil', 'dolar', 'inflacao', 'pib']

# --- Processamento Multi-Horizonte ---
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
modelos_finais = {}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, target) in enumerate(horizontes.items()):
    with tabs[i]:
        df_h = df_full.dropna(subset=[target])
        if len(df_h) < 48:
            st.warning("Dados insuficientes.")
            continue
            
        X, y = df_h[features], df_h[target]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Backtest
        preds_bt, actuals_bt = [], []
        for j in range(48, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            mdl = Ridge(alpha=0.5).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        res_bt = pd.Series(preds_bt, index=y.index[48:])
        mdl_f = Ridge(alpha=0.5).fit(X_s, y)
        modelos_finais[label] = (mdl_f, scaler)
        pred_u = mdl_f.predict(scaler.transform([u_inputs]))[0]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y[48:].cumsum(), label="Real", color="black", alpha=0.5)
            ax.plot(res_bt.cumsum(), label="Modelo", color="#1f77b4", ls="--")
            ax.set_title(f"Backtest Acumulado - {label}")
            ax.legend()
            st.pyplot(fig)
        with c2:
            st.metric("Retorno Projetado", f"{pred_u:.2%}")
            st.metric("Preço Alvo", f"{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")
            st.write(f"Margem Erro (IC 95%): ±{1.96*res_bt.std():.2%}")

# --- Tabela de Sensibilidade ---
st.divider()
st.header("3. Stress Test: Juros vs Dólar (Horizonte 1 Mês)")
if "1 Mês" in modelos_finais:
    mdl_s, scaler_s = modelos_finais["1 Mês"]
    # Variação de 0.50% Selic e 0.10 Dólar
    j_range = [u_inputs[0] + x for x in [-1.0, -0.5, 0, 0.5, 1.0]]
    d_range = [u_inputs[1] + x for x in [-0.2, -0.1, 0, 0.1, 0.2]]
    
    matrix = [[mdl_s.predict(scaler_s.transform([[j, d, u_inputs[2], u_inputs[3]]]))[0] for d in d_range] for j in j_range]
    df_sens = pd.DataFrame(matrix, index=[f"Selic {x:.2f}%" for x in j_range], columns=[f"Dólar R${x:.2f}" for x in d_range])
    
    st.dataframe(df_sens.style.format("{:.2%Short}").background_gradient(cmap="RdYlGn", axis=None))

# --- Base de Dados ---
st.divider()
st.header("4. Base de Dados Bruta (Valores Mensais Reais)")
st.dataframe(df_full.tail(15))
