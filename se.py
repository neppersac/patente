import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from scipy.stats import jarque_bera
import warnings
warnings.filterwarnings("ignore")

# Dados fornecidos
data = {
    'ano': list(range(1990, 2022)),
    'industria': [165748, 173335, 119602, 137954, 133632, 129840, 147873, 172890, 202909, 229719, 250739, 241367, 233223, 224842, 231643, 232446, 233655, 252360, 281970, 240522, 261072, 292761, 282678, 277978, 229953, 240178, 230826, 201378, 123435, 136023, 234305, 175275],
    'residuos': [443991, 466103, 498166, 529223, 564878, 604413, 647357, 694326, 745910, 802362, 868364, 943232, 1046692, 1178731, 1231369, 1271401, 1341265, 1474938, 1751928, 1870073, 1990357, 2001427, 1584782, 1701111, 1689370, 1743752, 1906731, 1852917, 1901295, 1890875, 1929675, 1938633],
    'energia': [1658841, 2139581, 2748061, 2127341, 2280753, 3067810, 3236794, 3749342, 5011432, 4530191, 4476912, 4834213, 5174105, 5246602, 6044931, 6705587, 6899836, 7793635, 7987983, 7943082, 9491642, 8690559, 8981528, 7933636, 10053720, 9052963, 7601955, 6871618, 8732482, 9315481, 8817008, 8750817],
    'agropecuaria': [1695250, 1717488, 1691897, 1824539, 1971429, 2117797, 1883953, 1984823, 2102121, 2169370, 2212781, 2276678, 2325104, 2798041, 2870780, 2965834, 3038900, 2773589, 3035828, 3123784, 3165481, 3297551, 3323456, 3343939, 3216791, 2988903, 3052084, 3123103, 3203261, 3361296, 3330574, 3508736],
    'mutf': [17997441, 20742183, 35379107, 35767011, 33234297, 38719384, 44443642, 67801401, 70949798, 40609582, 39656855, 31037525, 44898980, 76654850, 64953345, 69580132, 62746189, 43690816, 34959052, 37494546, 46813828, 29663960, 34169836, 47845600, 44143001, 70593534, 108383808, 64540122, 65635103, 105183535, 127373987, 124471910],
    'total': [21961271, 25238690, 40436833, 40386068, 38184989, 44639244, 50359619, 74402782, 79012169, 48341223, 47465651, 39333015, 53678103, 86103066, 75332067, 80755401, 74259844, 55985338, 48016762, 50672007, 61722380, 43946258, 48342280, 61102264, 59332835, 84619330, 121175403, 76589138, 79595577, 119887210, 141685548, 138845371]
}

# Função para selecionar o melhor modelo ETS
def melhor_ets(serie, treino, teste, modelos, titulo):
    melhor_modelo = None
    menor_aic = float('inf')
    menor_bic = float('inf')
    previsoes_dict = {}

    print(f"\n{titulo}")
    for modelo_nome, (error_type, trend_type, damped) in modelos.items():
        try:
            # Configurar o modelo ETS
            modelo = ETSModel(
                treino,
                error=error_type,
                trend=trend_type,
                damped_trend=damped,
                seasonal=None  # Sem sazonalidade
            )
            fitted_model = modelo.fit()

            # Extrair parâmetros diretamente do fitted_model
            param_names = fitted_model.param_names  # Lista de nomes dos parâmetros
            param_values = fitted_model.params      # Valores dos parâmetros como array
            params_dict = dict(zip(param_names, param_values))  # Converter para dicionário

            alpha = params_dict.get('smoothing_level', np.nan)
            beta = params_dict.get('smoothing_trend', np.nan)
            phi = params_dict.get('damping_trend', np.nan) if damped else np.nan
            l = params_dict.get('initial_level', np.nan)
            b = params_dict.get('initial_trend', np.nan)

            # Métricas de ajuste
            aic = fitted_model.aic
            bic = fitted_model.bic
            residuos = fitted_model.resid

            # Teste Jarque-Bera (normalidade dos resíduos)
            jb_stat, jb_pvalue = jarque_bera(residuos)

            # Previsões
            previsoes = fitted_model.forecast(steps=len(teste))
            rmse = np.sqrt(np.mean((teste - previsoes) ** 2))
            mae = np.mean(np.abs(teste - previsoes))
            mape = np.mean(np.abs((teste - previsoes) / teste)) * 100
              
            # Atualizar o melhor modelo com base no AIC
            if aic < menor_aic and jb_pvalue > 0:
                melhor_modelo = fitted_model
                menor_aic = aic
                menor_bic = bic 
                nome_melhor_modelo = modelo_nome       
            
            # Armazenar previsões
            previsoes_dict[modelo_nome] = previsoes

            # Exibir resultados
            # print(f"Modelo: {modelo_nome}, alpha: {alpha:.4f}, beta: {beta:.4f}, phi: {phi:.4f}, "
            #      f"AIC: {aic:.2f}, BIC: {bic:.2f}, JB p-value: {jb_pvalue:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

        except Exception as e:
            print(f"Erro no modelo {modelo_nome}: {e}")           

    return melhor_modelo, nome_melhor_modelo

# Configurações dos modelos
modelos = {
    'AA': ('add', 'add', False),    # Tendência aditiva, erro aditivo
    'MA': ('mul', 'add', False),    # Tendência aditiva, erro multiplicativo
    'MM': ('mul', 'mul', False),    # Tendência multiplicativa, erro multiplicativo
    'AAd': ('add', 'add', True),    # Tendência aditiva amortecida, erro aditivo
    'MAd': ('mul', 'add', True),    # Tendência aditiva amortecida, erro multiplicativo
    'MMd': ('mul', 'mul', True)     # Tendência multiplicativa amortecida, erro multiplicativo
}

# Divisão treino/teste (80% treino, 20% teste)
df = pd.DataFrame(data)
for col in df.columns[1:]:  # Ignorar a coluna 'ano'
    serie = df[col].values
    n = len(serie)
    treino_size = int(n * 0.8)
    treino, teste = serie[:treino_size], serie[treino_size:]

    # Aplicar a função
    melhor_modelo, modelo = melhor_ets(serie, treino, teste, modelos, col)
    
    if melhor_modelo:
        print(f"Melhor modelo para {col}: modelo = {modelo}, AIC = {melhor_modelo.aic:.2f}, BIC = {melhor_modelo.bic:.2f}")