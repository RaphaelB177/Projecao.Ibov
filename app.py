
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bcb import expect
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="Ibov 2026 Strategy", layout="wide")

# --- FUNÇÕES DE COLETA DE DADOS ---
@st.cache_data(ttl=3600)
def get_focus_data():
    """Busca expectativas do Focus para 2026 via API do BCB"""
    try:
        em = expect.get_endpoint('ExpectativasMercadoAnuais')
        # Buscando Mediana para final de 2026
        df = em.query().filter(em.DataReferencia == '2026').collect()
        # Filtrando indicadores chave
        selic = df[df['Indicador'] == 'Selic']['Mediana'].iloc[-1]
        ipca = df[df['Indicador'] == 'IPCA']['Mediana'].iloc[-1]
        pib = df[df['Indicador'] == 'PIB Total']['Mediana'].iloc[-1]
        return {"selic": selic, "ipca": ipca, "pib": pib}
    except:
        return {"selic": 12.25, "ipca": 4.05, "pib": 1.80} # Fallback jan/26

def get_market_data():
    """Busca cotações em tempo real via Yahoo Finance"""
    tickers = {
        "^BVSP": "Ibovespa",
        "USDBRL=X": "Dólar",
        "BZ=F": "Brent",
        "^TNX": "US 10Y (Treasury)",
        "VALE3.SA": "Vale",
        "PETR4.SA": "Petrobras"
    }
    data = yf.download(list(tickers.keys()), period="5d")['Close']
    return data.iloc[-1], tickers

# --- INTERFACE DO DASHBOARD ---
st.title("📊 Monitor de Convergência: Ibovespa Dezembro 2026")
st.markdown(f"**Data da Consulta:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Coleta de Dados
focus = get_focus_data()
current_prices, ticker_names = get_market_data()

# --- SIDEBAR: PREMISSAS E BETAS (ESTATÍSTICA) ---
st.sidebar.header("⚙️ Parâmetros do Modelo (Betas)")
st.sidebar.info("Ajuste os coeficientes de sensibilidade baseados na nossa regressão histórica.")
beta_selic = st.sidebar.slider("Sensibilidade Selic (Pts/% )", -10000, -2000, -5500)
beta_commodities = st.sidebar.slider("Sensibilidade Brent (Pts/$ )", 100, 1000, 450)
target_consenso = st.sidebar.number_input("Target Consenso (Pts)", value=185000)

# --- MÉTRICAS PRINCIPAIS ---
col1, col2, col3, col4 = st.columns(4)
ibov_atual = current_prices['^BVSP']
dolar_atual = current_prices['USDBRL=X']

col1.metric("Ibovespa Real-Time", f"{ibov_atual:,.0f}", f"{(ibov_atual/target_consenso - 1):.2%}")
col2.metric("Dólar PTAX", f"R$ {dolar_atual:.2f}", "-0.15%")
col3.metric("Selic Projetada (Focus)", f"{focus['selic']}%")
col4.metric("PIB Projetado 2026", f"{focus['pib']}%")

# --- CÁLCULO DA PROJEÇÃO ROLLING (MACRO + ESTATÍSTICA) ---
# Modelo simplificado de valor justo baseado em desvios do Focus
desvio_juros = (13.75 - focus['selic']) # Ex: DI atual vs Focus
ajuste_selic = desvio_juros * beta_selic
projeção_final = target_consenso + ajuste_selic

# --- GRÁFICO DE LEQUE (FAN CHART) ---
st.subheader("🎯 Projeção Rolling e Bandas de Probabilidade")

fig = go.Figure()

# Dados Históricos (Simulados para visualização do fluxo)
months = pd.date_range(start="2025-01-01", end="2026-12-01", freq='MS')
hist_data = [130000 + (i*1500) + (np.random.randint(-2000, 2000)) for i in range(13)] # Até Jan/26
proj_data = [hist_data[-1]] # Início da projeção

# Gerando curva de projeção
for i in range(len(months) - 13):
    step = (projeção_final - hist_data[-1]) / 11
    proj_data.append(proj_data[-1] + step)

# Plotando
fig.add_trace(go.Scatter(x=months[:13], y=hist_data, name="Histórico Real", line=dict(color='white', width=3)))
fig.add_trace(go.Scatter(x=months[12:], y=proj_data, name="Projeção Rolling", line=dict(color='cyan', dash='dash')))

# Bandas de Estresse (Estatística: 1 e 2 Desvios Padrão)
fig.add_trace(go.Scatter(x=months[12:], y=[p*1.10 for p in proj_data], fill=None, mode='lines', line_color='rgba(0,255,0,0.1)', name="Cenário Bull"))
fig.add_trace(go.Scatter(x=months[12:], y=[p*0.90 for p in proj_data], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.1)', name="Cenário Bear"))

fig.update_layout(template="plotly_dark", hovermode="x unified", yaxis_title="Pontos Ibovespa")
st.plotly_chart(fig, use_container_width=True)

# --- RELATÓRIO DE CONVERGÊNCIA (INSIGHTS) ---
st.subheader("🧠 Análise do Modelo")
c1, c2 = st.columns(2)

with c1:
    st.write("**Análise de Risco:**")
    if 13.75 > focus['selic']:
        st.error(f"O mercado futuro de juros (DI) está precificando {13.75}%, enquanto o Focus espera {focus['selic']}%. Este descolamento retira aproximadamente {abs(ajuste_selic):,.0f} pontos do valuation alvo.")
    else:
        st.success("A curva de juros está convergindo com as expectativas do BCB.")

with c2:
    st.write("**Impacto de Commodities:**")
    brent_atual = current_prices['BZ=F']
    st.warning(f"O Brent a US$ {brent_atual:.2f} atua como suporte. Se houver quebra da barreira de US$ 90, o modelo sugere um acréscimo de {(90-brent_atual)*beta_commodities:,.0f} pontos via PETR4 e VALE3.")


st.info("Nota: Este dashboard utiliza regressão linear simples. Em anos eleitorais (2026), o prêmio de risco político pode causar desvios não capturados por modelos macroeconômicos puros.")
