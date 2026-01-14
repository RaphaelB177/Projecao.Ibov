import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Configurações iniciais
st.set_page_config(layout="wide", page_title="Ibov Projeção Macro", page_icon="📈")

# --- LÓGICA DE REFRESH MANUAL ---
st.sidebar.title("Configurações")
if st.sidebar.button("🔄 Forçar Atualização (API Refresh)"):
    st.cache_data.clear()
    st.rerun()

# --- FUNÇÃO DE DOWNLOAD ROBUSTA (CORREÇÃO KEYERROR) ---
def download_yf_safe(ticker, start_date):
    try:
        # Baixa os dados sem progresso para não sujar o log
        data = yf.download(ticker, start=start_date, progress=False)
        
        if data.empty:
            st.error(f"⚠️ O Yahoo Finance retornou um conjunto de dados vazio para {ticker}.")
            return pd.Series()

        # Tratamento de Multi-Index ou Colunas Simples
        # O yfinance mudou recentemente para MultiIndex por padrão
        if isinstance(data.columns, pd.MultiIndex):
            # Tenta Adj Close, senão vai de Close
            if 'Adj Close' in data.columns.get_level_values(0):
                return data['Adj Close'][ticker]
            if 'Close' in data.columns.get_level_values(0):
                return data['Close'][ticker]
        else:
            # Caso o DataFrame venha simples
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            if 'Close' in data.columns:
                return data['Close']

        # Se chegar aqui, as colunas mudaram de nome novamente
        st.error(f"❌ Erro de estrutura no Yahoo: Colunas encontradas: {data.columns.tolist()}")
        return pd.Series()
        
    except Exception as e:
        st.error(f"❌ Erro Crítico no Yahoo Finance ({ticker}): {type(e).__name__} - {str(e)}")
        return pd.Series()

# --- FUNÇÃO FRED VIA CSV DIRETO ---
def get_fred_data(series_code, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    try:
        df_fred = pd.read_csv(url, index_col='DATE', parse_dates=True)
        return df_fred[df_fred.index >= start_date]
    except Exception as e:
        st.warning(f"⚠️ Não foi possível obter dados do FRED: {str(e)}")
        return pd.DataFrame()

# -------------------
# CARREGAMENTO COM CACHE (Só roda se clicar no Refresh)
# -------------------
@st.cache_data(ttl=None)
def load_data():
    start_date = "2010-01-01"
    
    with st.spinner("📦 Sincronizando com bases de dados (Yahoo, BCB, FRED)..."):
        # 1. Ibovespa
        ibov = download_yf_safe("^BVSP", start_date)
        if ibov.empty:
            st.info("Dica: Se o erro for Rate Limit, aguarde alguns minutos antes de dar Refresh.")
            st.stop()
        ibov = ibov.resample('ME').last()

        # 2. Dados via Banco Central (SGS) - Muito mais estável que Yahoo para Dólar
        try:
            # 1: Dólar Venda, 433: IPCA, 4390: SELIC, 438: PIB
            dict_sgs = {'dolar': 1, 'inflacao': 433, 'juros_brasil': 4390, 'pib': 438}
            df_sgs = sgs.get(dict_sgs, start=start_date)
        except Exception as e:
            st.error(f"❌ Erro na API do Banco Central: {str(e)}")
            st.stop()

        # 3. Juros Americanos (FRED)
        juros_usa = get_fred_data('FEDFUNDS', start_date)
        juros_usa.columns = ['juros_americano']

        # Consolidação e Limpeza
        main_df = pd.DataFrame(index=ibov.index)
        main_df['ibov'] = ibov
        
        for d in [df_sgs, juros_usa]:
            d.index = pd.to_datetime(d.index)
            # Unimos garantindo que as datas batam no fechamento do mês
            main_df = main_df.join(d.resample('ME').last(), how='left')

        # Preenche vazios (ffill) e cria o alvo (retorno do próximo mês)
        main_df = main_df.ffill().dropna()
        main_df['target_ret'] = main_df['ibov'].pct_change().shift(-1)
        
        return main_df.dropna()

# Início da Execução
df = load_data()

# -------------------
# MODELAGEM E DASHBOARD
# -------------------
st.title("📈 Projeção Ex-Ante Ibovespa")
st.markdown(f"**Status dos Dados:** Atualizado até `{df.index[-1].strftime('%m/%Y')}`")

# Definição das variáveis
features = ["juros_brasil", "dolar", "pib", "inflacao", "juros_americano"]
X = df[features]
y = df["target_ret"]

# Normalização e Treino
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# Sidebar para Simulação
st.sidebar.divider()
st.sidebar.subheader("Cenário para Próximo Mês")
user_inputs = []
friendly_names = ["Selic Brasil (%)", "Dólar (R$)", "PIB (Var %)", "IPCA (%)", "Juros EUA (%)"]

for i, f in enumerate(features):
    val = st.sidebar.number_input(
        friendly_names[i], 
        value=float(df[f].iloc[-1]), 
        format="%.2f",
        help=f"Valor atual extraído da base: {df[f].iloc[-1]:.2f}"
    )
    user_inputs.append(val)

# Cálculo da Projeção
user_data_scaled = scaler.transform([user_inputs])
pred_retorno = model.predict(user_data_scaled)[0]
preco_atual = df['ibov'].iloc[-1]
preco_projetado = preco_atual * (1 + pred_retorno)

# Exibição de Resultados
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Projeção Retorno (1M)", f"{pred_retorno:.2%}")
with c2:
    st.metric("Ibovespa Alvo", f"{preco_projetado:,.0f}")
with c3:
    st.metric("Aderência do Modelo (R²)", f"{model.score(X_scaled, y):.2f}")

# Gráficos
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Importância de cada Indicador")
    st.caption("Pesos normalizados do modelo Ridge Regression")
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    colors = ['#ff4b4b' if x < 0 else '#00cc96' for x in model.coef_]
    importances = pd.Series(model.coef_, index=friendly_names).sort_values()
    importances.plot(kind='barh', color=colors, ax=ax_imp)
    ax_imp.axvline(0, color='black', lw=0.8)
    st.pyplot(fig_imp)

with col_right:
    st.subheader("📊 Histórico Ibovespa")
    st.line_chart(df['ibov'])

st.caption("Nota: Este modelo é educacional e utiliza regressão linear simples com penalidade Ridge.")
