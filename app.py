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
# 1. Configurações e Coleta de Dados
# -------------------
st.set_page_config(layout="wide", page_title="Master Ibov Dashboard", page_icon="📈")

@st.cache_data(ttl=None)
def load_data_master():
    hoje = datetime.now() - timedelta(days=2)
    # Aumentamos a janela para 15 anos para garantir volume de dados para horizontes longos
    start_str = (hoje - timedelta(days=365*15)).strftime('%Y-%m-%d')
    
    try:
        ibov_raw = yf.download("^BVSP", start=start_str, progress=False)
        if isinstance(ibov_raw.columns, pd.MultiIndex):
            col = 'Adj Close' if 'Adj Close' in ibov_raw.columns.get_level_values(0) else 'Close'
            ibov = ibov_raw[col].iloc[:, 0]
        else:
            col = 'Adj Close' if 'Adj Close' in ibov_raw.columns else 'Close'
            ibov = ibov_raw[col]
        df = ibov.resample('ME').last().to_frame('ibov')
    except:
        st.error("Falha ao conectar com Yahoo Finance.")
        st.stop()

    # Fallbacks baseados em dados de Janeiro/2026
    fallbacks = {'dolar': 5.37, 'inflacao': 4.4, 'juros_brasil': 15.0, 'pib': 3.0}

    def get_sgs(codigo, nome):
        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv"
            r = requests.get(url, timeout=10)
            d = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',')
            d['data'] = pd.to_datetime(d['data'], dayfirst=True)
            return d.rename(columns={'valor': nome}).set_index('data').resample('ME').last()
        except:
            return pd.DataFrame()

    for cod, nome in [(1, 'dolar'), (433, 'inflacao'), (4390, 'juros_brasil'), (438, 'pib')]:
        sgs_df = get_sgs(cod, nome)
        if not sgs_df.empty:
            df = df.join(sgs_df, how='left')
        else:
            # Se falhar, preenche com o último valor global conhecido
            df[nome] = fallbacks[nome]
    
    df = df.ffill().bfill()
    
    # Criando os alvos para 1, 6 e 12 meses (Retornos Acumulados)
    df['ret_1m'] = df['ibov'].pct_change(1).shift(-1)
    df['ret_6m'] = df['ibov'].pct_change(6).shift(-6)
    df['ret_12m'] = df['ibov'].pct_change(12).shift(-12)
    return df

df_full = load_data_master()

# -------------------
# 2. Sidebar e Inputs
# -------------------
st.sidebar.title("🛠️ Configurações")
window_type = st.sidebar.selectbox("Tipo de Janela Backtest:", ["Expanding", "Rolling"])
rolling_size = st.sidebar.slider("Tamanho da Rolling Window (meses):", 12, 60, 36)

st.sidebar.divider()
st.sidebar.header("🔮 Cenário Futuro")
features = ['juros_brasil', 'dolar', 'inflacao', 'pib']
u_inputs = []
for f in features:
    val = st.sidebar.number_input(f"Expectativa {f}", value=float(df_full[f].iloc[-1]), format="%.2f")
    u_inputs.append(val)

# -------------------
# 3. Análise de Correlação
# -------------------
st.title("📊 Projeção Estatística do Ibovespa")
st.header("1. Correlação Macro vs Retornos")
corr_target = df_full[features + ['ret_1m', 'ret_6m', 'ret_12m']].corr().loc[features, ['ret_1m', 'ret_6m', 'ret_12m']]
fig_corr, ax_corr = plt.subplots(figsize=(10, 3))
sns.heatmap(corr_target.T, annot=True, cmap="RdYlGn", center=0, ax=ax_corr, fmt=".2f")
st.pyplot(fig_corr)


# -------------------
# 4. Processamento Multi-Horizonte
# -------------------
st.header("2. Projeções e Backtest")
horizontes = {"1 Mês": "ret_1m", "6 Meses": "ret_6m", "12 Meses": "ret_12m"}
alphas = {"1 Mês": 0.5, "6 Meses": 0.5, "12 Meses": 1.0}
modelos_finais = {}

tabs = st.tabs(list(horizontes.keys()))

for i, (label, target) in enumerate(horizontes.items()):
    with tabs[i]:
        # Filtragem rigorosa para evitar o ValueError
        df_h = df_full.dropna(subset=[target])
        
        if len(df_h) < 60:
            st.warning(f"Dados insuficientes para calcular o horizonte {label}.")
            continue
            
        X = df_h[features]
        y = df_h[target]
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X) # Aqui o erro era disparado; agora X_s está garantido
        
        # Backtest (Lógica Original Recuperada)
        preds_bt, actuals_bt = [], []
        start_idx = 48 # Ponto de partida
        for j in range(start_idx, len(y)):
            X_t = X_s[:j] if window_type == "Expanding" else X_s[max(0, j-rolling_size):j]
            y_t = y[:j] if window_type == "Expanding" else y[max(0, j-rolling_size):j]
            mdl = Ridge(alpha=alphas[label]).fit(X_t, y_t)
            preds_bt.append(mdl.predict(X_s[j:j+1])[0])
            actuals_bt.append(y.iloc[j])
        
        res_bt = pd.Series(preds_bt, index=y.index[start_idx:])
        mdl_f = Ridge(alpha=alphas[label]).fit(X_s, y)
        modelos_finais[label] = (mdl_f, scaler)
        pred_u = mdl_f.predict(scaler.transform([u_inputs]))[0]

        # Gráfico de Performance
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
            # Cálculo de Intervalo de Confiança dinâmico do código original
            err_std = (y[start_idx:] - res_bt).std()
            ax_bt.plot(y[start_idx:].cumsum(), label="Real (Acumulado)", color="black", lw=1.5)
            ax_bt.plot(res_bt.cumsum(), label="Modelo (Acumulado)", color="#1f77b4", ls="--")
            ax_bt.fill_between(res_bt.index, (res_bt - 1.96*err_std).cumsum(), (res_bt + 1.96*err_std).cumsum(), 
                               color='#1f77b4', alpha=0.1, label="IC 95%")
            ax_bt.legend()
            st.pyplot(fig_bt)
            
        
        with c2:
            st.metric("Retorno Projetado", f"{pred_u:.2%}")
            st.metric("Preço Alvo", f"{df_full['ibov'].iloc[-1]*(1+pred_u):,.0f}")
            st.write(f"Incerteza do Modelo (RMSE): {np.sqrt(mean_squared_error(y[start_idx:], res_bt)):.4f}")

# -------------------
# 5. Sensibilidade e Dados Brutos
# -------------------
st.divider()
st.header("3. Análise de Sensibilidade (Stress Test)")
if "1 Mês" in modelos_finais:
    mdl_s, scaler_s = modelos_finais["1 Mês"]
    j_range = np.linspace(u_inputs[0]*0.9, u_inputs[0]*1.1, 7) # Var de +/- 10%
    d_range = np.linspace(u_inputs[1]*0.9, u_inputs[1]*1.1, 7)
    
    sens_matrix = [[mdl_s.predict(scaler_s.transform([[j, d, u_inputs[2], u_inputs[3]]]))[0] for d in d_range] for j in j_range]
    df_sens = pd.DataFrame(sens_matrix, index=[f"Selic {x:.2f}%" for x in j_range], columns=[f"Dólar R${x:.2f}" for x in d_range])
    
    st.dataframe(df_sens.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=None))
    

st.divider()
st.header("4. Base de Dados Processada")
st.dataframe(df_full.tail(15))
