import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Ibov Projeção Macro")

# --- Função para baixar FRED sem pandas-datareader ---
def get_fred_data(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    df_fred = pd.read_csv(url, index_col='DATE', parse_dates=True)
    return df_fred[df_fred.index >= start_date]

# -------------------
# Carregamento de Dados Otimizado
# -------------------
@st.cache_data(ttl=3600)
def load_data():
    start_date = "2010-01-01"
    try:
        # Ibov e Dólar via Yahoo
        ibov = yf.download("^BVSP", start=start_date)['Adj Close'].resample('ME').last()
        dolar = yf.download("USDBRL=X", start=start_date)['Adj Close'].resample('ME').last()
        
        # Dados Brasil via BCB (SGS)
        ipca = sgs.get({'inflacao': 433}, start=start_date)
        selic = sgs.get({'juros_brasil': 4390}, start=start_date)
        pib = sgs.get({'pib': 438}, start=start_date)
        
        # Juros USA via FRED (URL Direta para evitar erro de distutils)
        juros_usa = get_fred_data('FEDFUNDS', start_date)
        juros_usa.columns = ['juros_americano']
        
        # Consolidação
        main_df = pd.DataFrame(index=ibov.index)
        main_df['ibov'] = ibov
        main_df['dolar'] = dolar
        
        for d in [selic, ipca, pib, juros_usa]:
            d.index = pd.to_datetime(d.index)
            main_df = main_df.join(d.resample('ME').last(), how='left')
        
        main_df = main_df.ffill().dropna()
        main_df['target_ret'] = main_df['ibov'].pct_change().shift(-1)
        return main_df.dropna()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.info("Conectando às fontes de dados financeiras...")
    st.stop()

# -------------------
# Interface e Modelo
# -------------------
st.title("📈 Projeção Ibovespa (Ex-Ante)")

features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo final
model = Ridge(alpha=1.0).fit(X_scaled, y)

# Inputs do Usuário
st.sidebar.header("Cenário de Projeção")
vals = []
for f in features:
    default_val = float(df[f].iloc[-1])
    vals.append(st.sidebar.number_input(f"Valor para {f}", value=default_val))

# Predição
user_scaled = scaler.transform([vals])
pred_ret = model.predict(user_scaled)[0]

col1, col2 = st.columns(2)
with col1:
    st.metric("Projeção Retorno (Próx. Mês)", f"{pred_ret:.2%}")
    st.metric("Preço Estimado", f"{df['ibov'].iloc[-1] * (1+pred_ret):,.0f}")

with col2:
    # Gráfico de Importância
    fig, ax = plt.subplots()
    pd.Series(model.coef_, index=features).plot(kind='barh', ax=ax, color='teal')
    ax.set_title("Importância das Variáveis")
    st.pyplot(fig)

st.divider()
st.subheader("Histórico de Fechamento vs Projeção")
st.line_chart(df['ibov'])
