import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="Análise de Emissões CO2e", layout="wide")

# Dados das séries temporais
data = {
    'ano': list(range(1990, 2022)),
    'industria': [165748, 173335, 119602, 137954, 133632, 129840, 147873, 172890, 202909, 229719, 250739, 241367, 233223, 224842, 231643, 232446, 233655, 252360, 281970, 240522, 261072, 292761, 282678, 277978, 229953, 240178, 230826, 201378, 123435, 136023, 234305, 175275],
    'residuos': [443991, 466103, 498166, 529223, 564878, 604413, 647357, 694326, 745910, 802362, 868364, 943232, 1046692, 1178731, 1231369, 1271401, 1341265, 1474938, 1751928, 1870073, 1990357, 2001427, 1584782, 1701111, 1689370, 1743752, 1906731, 1852917, 1901295, 1890875, 1929675, 1938633],
    'energia': [1658841, 2139581, 2748061, 2127341, 2280753, 3067810, 3236794, 3749342, 5011432, 4530191, 4476912, 4834213, 5174105, 5246602, 6044931, 6705587, 6899836, 7793635, 7987983, 7943082, 9491642, 8690559, 8981528, 7933636, 10053720, 9052963, 7601955, 6871618, 8732482, 9315481, 8817008, 8750817],
    'agropecuaria': [1695250, 1717488, 1691897, 1824539, 1971429, 2117797, 1883953, 1984823, 2102121, 2169370, 2212781, 2276678, 2325104, 2798041, 2870780, 2965834, 3038900, 2773589, 3035828, 3123784, 3165481, 3297551, 3323456, 3343939, 3216791, 2988903, 3052084, 3123103, 3203261, 3361296, 3330574, 3508736],
    'mutf': [17997441, 20742183, 35379107, 35767011, 33234297, 38719384, 44443642, 67801401, 70949798, 40609582, 39656855, 31037525, 44898980, 76654850, 64953345, 69580132, 62746189, 43690816, 34959052, 37494546, 46813828, 29663960, 34169836, 47845600, 44143001, 70593534, 108383808, 64540122, 65635103, 105183535, 127373987, 124471910],
    'total': [21961271, 25238690, 40436833, 40386068, 38184989, 44639244, 50359619, 74402782, 79012169, 48341223, 47465651, 39333015, 53678103, 86103066, 75332067, 80755401, 74259844, 55985338, 48016762, 50672007, 61722380, 43946258, 48342280, 61102264, 59332835, 84619330, 121175403, 76589138, 79595577, 119887210, 141685548, 138845371]
}

df = pd.DataFrame(data)

# Título da aplicação
st.title('Análise de Emissões de CO2e (1990-2021)')
st.markdown("""
Esta aplicação analisa as séries temporais de emissões de gases de efeito estufa (em CO2 equivalente) para diferentes setores.
""")

# Sidebar
st.sidebar.header('Configurações')
setor = st.sidebar.selectbox('Selecione o setor:', ['total', 'industria', 'residuos', 'energia', 'agropecuaria', 'mutf'])
anos_previsao = st.sidebar.slider('Anos para previsão:', 1, 10, 5)

# Função para análise ARIMA
def analisar_arima(serie, anos_previsao):
    # Treinar modelo ARIMA
    model = ARIMA(serie, order=(1,1,1))
    model_fit = model.fit()
    
    # Previsão
    forecast = model_fit.get_forecast(steps=anos_previsao)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    return forecast_mean, conf_int, model_fit

# Visualização dos dados
st.header(f'Série Temporal - {setor.upper()}')
fig = px.line(df, x='ano', y=setor, title=f'Emissões de CO2e - {setor.upper()}')
st.plotly_chart(fig, use_container_width=True)

# Análise ARIMA
st.header('Análise e Previsão ARIMA')

# Gráficos ACF e PACF
st.subheader('Autocorrelação (ACF e PACF)')
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    plot_acf(df[setor], ax=ax)
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    plot_pacf(df[setor], ax=ax)
    st.pyplot(fig)

# Modelagem e previsão
try:
    forecast_mean, conf_int, model_fit = analisar_arima(df[setor], anos_previsao)
    
    # Resultados do modelo
    st.subheader('Resultados do Modelo ARIMA')
    st.text(model_fit.summary())
    
    # Gráfico de previsão
    st.subheader(f'Previsão para os próximos {anos_previsao} anos')
    
    # Criar DataFrame para visualização
    ultimo_ano = df['ano'].iloc[-1]
    anos_futuros = list(range(ultimo_ano + 1, ultimo_ano + 1 + anos_previsao))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['ano'], df[setor], label='Dados históricos')
    ax.plot(anos_futuros, forecast_mean, label='Previsão', color='red')
    ax.fill_between(anos_futuros, 
                   conf_int.iloc[:, 0], 
                   conf_int.iloc[:, 1], 
                   color='pink', alpha=0.3, label='Intervalo de confiança')
    ax.set_title(f'Previsão de Emissões para {setor.upper()}')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Emissões CO2e')
    ax.legend()
    st.pyplot(fig)
    
    # Mostrar valores previstos
    st.subheader('Valores Previstos')
    prev_df = pd.DataFrame({
        'Ano': anos_futuros,
        'Previsão': forecast_mean,
        'Limite Inferior': conf_int.iloc[:, 0],
        'Limite Superior': conf_int.iloc[:, 1]
    })
    st.dataframe(prev_df.style.format("{:,.0f}"))
    
except Exception as e:
    st.error(f"Erro ao ajustar modelo ARIMA: {str(e)}")

# Comparação entre setores
st.header('Comparação entre Setores')
setores_comparacao = st.multiselect(
    'Selecione setores para comparar:',
    ['industria', 'residuos', 'energia', 'agropecuaria', 'mutf', 'total'],
    default=['total', 'energia', 'agropecuaria']
)

if setores_comparacao:
    fig = px.line(df, x='ano', y=setores_comparacao, 
                 title='Comparação entre Setores',
                 labels={'value': 'Emissões CO2e', 'variable': 'Setor'})
    st.plotly_chart(fig, use_container_width=True)

# Rodapé
st.markdown("""
---
**Fonte dos dados**: Séries temporais anuais de emissões de CO2e (1990-2021)
""")