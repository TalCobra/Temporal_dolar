# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:22:12 2024

@author: Matheus Henrique Dizaro Miyamoto
"""
#%%
from sklearn.metrics import mean_absolute_percentage_error as mape
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# !pip install pmdarima
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import SimpleExpSmoothing
from scipy.stats import kstest
import pmdarima as pm
from scipy import stats
# !pip install arch
from arch import arch_model
import seaborn as sns
# !pip install python-bcb
from bcb import sgs
import statsmodels.api as sm
import yfinance as yf
#%%Funcao para baixar dados do Yahoo Finance (Selecionar todos os comandos)
def obter_dados(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

#%%Funcao para plotar os 4 graficos (Selecionar todos os comandos)
def plotar_graficos(data):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # Grafico 1: Abertura
    axs[0].plot(data['Open'], label='Abertura', color='blue')
    axs[0].set_ylabel('Abertura')

    # Grafico 2: Mi­nima
    axs[1].plot(data['Low'], label='Mi­nima', color='green')
    axs[1].set_ylabel('Minima')

    # Grafico 3: Maxima
    axs[2].plot(data['High'], label='Maxima', color='red')
    axs[2].set_ylabel('Maxima')

    # Grafico 4: Fechamento
    axs[3].plot(data['Close'], label='Fechamento', color='purple')
    axs[3].set_ylabel('Fechamento')

    axs[3].set_xlabel('Data')

    # Adiciona legenda
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    plt.tight_layout()
    plt.show()
#%%
ticker = "USDBRL=X"

start_date = "2003-01-01"
end_date = "2024-11-10"


#%%
dados = obter_dados(ticker, start_date, end_date)
plotar_graficos(dados)

dados_utilizar = dados
dados_utilizar .drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

treino = dados_utilizar[:'2024-09-30']
teste = dados_utilizar['2024-10-01':]


fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(dados_utilizar, lags=24, ax=axes[0])
plot_pacf(dados_utilizar, lags=24, ax=axes[1], method='ywm')
plt.show()

#%%Teste de Dickey Fuller
result = adfuller(dados_utilizar.dropna())
print(f'Resultado do Teste ADF: p-valor = {result[1]}')
if result[1] < 0.05:
    print("A série é estacionária.")
else:
    print("A série não é estacionária.")
#%%
def verificar_differenciacao(serie, nome):
    # Usar a função ndiffs do pmdarima
    d = pm.arima.ndiffs(serie, test='adf')  # Teste de Dickey-Fuller aumentado
    print(f"A série {nome} precisa de {d} diferenciação(ões) para ser estacionária.")
    return d

verificar_differenciacao(dados_utilizar, "Dolar - Treinamento")

# Diferenciação para estacionariedade
dolar_diff = dados_utilizar.diff().dropna()

fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(dolar_diff, lags=24, ax=axes[0])
plot_pacf(dolar_diff, lags=24, ax=axes[1], method='ywm')
plt.show()

#Naive
naive_forecast = pd.Series([treino.iloc[-1]] * len(teste), index=teste.index)
mape_naive = mape(teste, naive_forecast) * 100

#SES
ses_model = SimpleExpSmoothing(treino).fit(optimized=True)
ses_forecast = ses_model.forecast(steps=len(teste))
mape_ses = mape(teste, ses_forecast) * 100

residuos_ses = ses_model.resid
ljung_box = sm.stats.acorr_ljungbox(residuos_ses, lags=[10], return_df=True)
print(f'Resultado do teste de Ljung-Box:\n{ljung_box}')
#Como o p-value é maior que 0.5 os residuos não tem relação

ks_stat, p_value = kstest(ses_model.resid, 'norm', args=(np.mean(ses_model.resid), np.std(ses_model.resid)))
print(f'Teste de Kolmogorov-Smirnov para normalidade: p-valor = {p_value}')
if p_value > 0.01:
    print("Os resíduos seguem uma distribuição normal.")
else:
    print("Os resíduos não seguem uma distribuição normal.")
    
am = arch_model(residuos_ses, vol='ARCH', p=1)
test_arch = am.fit(disp='off')
print(test_arch.summary())
# não há efeitos ARCH pois p-value é maior que 0.05
#%%
n_periods = 5
index_of_fc = pd.date_range(treino.index[-1], periods=n_periods+1, freq='MS')[1:]
previsoes_diff = ses_model.forecast(steps=n_periods)
print(f"Previsões diferenciadas: {previsoes_diff}")

index_of_fc = pd.date_range(treino.index[-1], periods=n_periods+1, freq='MS')[1:]

ultimo_valor_original = treino.iloc[-1] # Último valor conhecido da série original (não diferenciada)
previsoes_nivel_original = [ultimo_valor_original]
print(ultimo_valor_original)
print(previsoes_nivel_original)

for previsao in previsoes_diff:
    novo_valor = previsoes_nivel_original[-1] + previsao
    previsoes_nivel_original.append(novo_valor)

previsoes_nivel_original = previsoes_nivel_original[1:]
print(previsoes_nivel_original)

previsoes_nivel_original_series = pd.Series(previsoes_nivel_original, index=index_of_fc)
print(previsoes_nivel_original_series)