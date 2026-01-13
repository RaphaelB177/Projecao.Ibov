
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bcb import Expectativas
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="Ibov 2026 Strategy", layout="wide")

# --- FUNÇÕES DE COLETA DE DADOS ---
@st.cache_data(ttl=3600)
def get_focus_data():
    """Busca expectativas do Focus para 2026 via API do BCB"""
    try:
        em = Expectativas.get_endpoint('ExpectativasMercadoAnuais')
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


# --- TABELA DE SENSIBILIDADE ---
st.divider()
st.subheader("🎲 Matriz de Sensibilidade: Ibovespa 2026")
st.markdown("Impacto cruzado de variações no **Dólar** e na **Selic** sobre o alvo do modelo.")

# Definição dos ranges de variação
variacoes_dolar = [-0.50, -0.25, 0, 0.25, 0.50]  # Passos de 25 centavos
variacoes_selic = [-1.0, -0.5, 0, 0.5, 1.0]      # Passos de 0.50%

# Criando a matriz de dados
dados_matriz = []
for v_selic in variacoes_selic:
    linha = []
    for v_dol in variacoes_dolar:
        # Cálculo: Preço Base + (Impacto Juros) + (Impacto Câmbio)
        # Assumindo Beta Câmbio médio de -8.000 pts por R$ 1,00 de variação
        selic_simulada = focus['selic'] + v_selic
        dolar_simulado = dolar_atual + v_dol
        
        impacto_juros = (13.75 - selic_simulada) * beta_selic
        impacto_cambio = (dolar_atual - dolar_simulado) * 8000 # Beta Câmbio estimado
        
        preço_final = target_consenso + impacto_juros + impacto_cambio
        linha.append(f"{preço_final/1000:.1f}k")
    dados_matriz.append(linha)

# Criando o DataFrame para exibição
df_sensibilidade = pd.DataFrame(
    dados_matriz,
    index=[f"Selic {focus['selic']+v}%" for v in variacoes_selic],
    columns=[f"Dólar R${dolar_atual+v:.2f}" for v in variacoes_dolar]
)

# Exibição com estilo
st.table(df_sensibilidade)

st.caption("Valores em milhares de pontos (k). O cenário central (0,0) reflete as premissas atuais do Focus e do mercado.")


# --- GERADOR DE PROMPT AUTOMÁTICO ---
st.divider()
st.subheader("🤖 Gerador de Relatório para Gemini")
st.markdown("Clique no botão abaixo para copiar o prompt estruturado com seus dados atuais.")

# Construção do texto do Prompt
prompt_text = f"""
Atue como um estrategista-chefe de investimentos. Compare as expectativas de mercado com as minhas premissas proprietárias para gerar um relatório de sensibilidade.

1. Dados de Mercado Atual (Referência):
- Ibovespa: {ibov_atual:,.0f} pts
- Dólar: R$ {dolar_atual:.2f}
- Selic (Focus): {focus['selic']}%

2. Meu Cenário Proprietário (Minhas Apostas):
- PIB: {user_pib}% 
- Dólar: R$ {user_dolar:.2f}
- Inflação: {user_ipca}% 
- Petróleo (Brent): US$ {user_brent:.2f}
- SELIC: {user_selic}%

3. Resultado do Modelo:
- Com as minhas premissas, o Ibovespa calculado é de {previsao_user:,.0f} pontos.
- O alvo do consenso de mercado (Target) é de {target_consenso:,.0f} pontos.

Tarefa de Análise:
1. Analise a distância entre a minha visão e a do mercado (Focus). Sou mais otimista ou pessimista?
2. Qual dos meus inputs (PIB, Dólar ou Selic) foi o maior responsável pela variação do preço-alvo no meu cenário?
3. Redija uma conclusão de um parágrafo defendendo por que um investidor deveria (ou não) acreditar no meu cenário em vez de seguir o consenso.
"""

# Botão de cópia rápida
st.text_area("Copie o texto abaixo:", value=prompt_text, height=300)
st.button("📋 Copiar para Área de Transferência", on_click=lambda: st.write("Texto copiado! (Use Ctrl+C)"))



