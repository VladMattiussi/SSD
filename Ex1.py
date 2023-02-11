# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:48:52 2022

@author: Vlad
"""

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import gamma
import scipy.stats as stats
from scipy.optimize import leastsq

#importo i dati
df = pd.read_csv("seriefit2021.csv")
plt.plot(df.y,label="Fit")
plt.grid()
plt.legend()
plt.show()

x = np.linspace(0, 65, 66)
y = df['y']

ds = df[df.columns[1]] # converto in serie

#decido di modellare la funzione di trend con una polinomiale
poly = np.polyfit(x, y, 3)
p = np.poly1d(poly)

plt.scatter(x, y, color='red')
plt.plot(x,y,color="blue")
plt.plot(x,p(x),color="green")
plt.show()

#analizzo le componenti di trend,seasonality e residuo
result = seasonal_decompose(ds, model='additive',period=12)
result.plot()
plt.show()

#Detrend
detrend = y - result.trend
plt.plot(x,result.trend,color="blue")
plt.plot(x,detrend, color="red")
plt.show()

result1 = seasonal_decompose(ds, model='multiplicative',period=12)
result1.plot()
plt.show()

#additive is better trend function

residuo = result.resid
stagione = result.seasonal

#ACF
diffdata = ds.diff()
diffdata[0] = ds[0] # reset 1st elem
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(diffdata, lags=36)
plt.show()

#residuo e seasonality
deseasoning = detrend - result.seasonal
plt.plot(x,deseasoning,color="orange")
plt.plot(x,result.seasonal,color="red")
plt.scatter(x, result.seasonal, color='black')
plt.show()

#guardando l'autocorrelazione il grafico sembra essere gia' destagionalizzato
###########################
#Confronto tra seasonality calcolata attraverso "seasonal_decompose"
#e quella attraverso calcoli manuali
test = y - p(x)
plt.plot(x,test,color="purple")
plt.plot(x,detrend,color="black")
plt.show()

#Confronto fra trend originale e funzione di trend
plt.plot(x,result.trend,color="yellow")
plt.plot(x,p(x),color="orange")
plt.show()
##########################

#Predizione seasonality
season_coeff = []
for d in [0,1,2,3,4,5,6]:
    season_coeff.append(np.mean(test[d::7]))

final_season = season_coeff[1:] 
final_season = np.append(final_season, season_coeff)
final_season = np.resize(final_season,36)


#Predizione per ulteriori 36 periodi di tempo
x1 = np.linspace(65, 101, 36)
preds = (p(x1)) + (final_season)
plt.plot(x,y,color="blue")
plt.plot(x1,preds,color="green")
plt.show()

#Prova di trovare la funzione di trend anche attraverso la distribuzione gamma 
def my_res( params, yData ):
    a, b = params
    xList= range( 1, len(yData) + 1 )
    th = np.fromiter( ( stats.gamma.pdf( x, a, loc=0, scale=b ) for x in  xList ), np.float )
    diff = th - np.array( yData )
    return diff

#making a least square fit with the pdf
sol, err = leastsq( my_res, [.4, 1 ], args=(y , ) )
datath = [ stats.gamma.pdf( z, sol[0], loc=0, scale=sol[1]) for z in range(0,65) ]

#the result gives the expected answer
plop=stats.gamma.pdf(x, sol[0], loc=0, scale=sol[1])

plt.plot(x, gamma.pdf(x, sol[0], loc=0, scale=sol[1]), label="fitted gamma")
plt.plot(y/180000,label="original")
plt.legend()
plt.show()

#generating random numbers with gamma distribution
testData = stats.gamma.rvs(sol[0], loc=0, scale=sol[1], size=5000 )
print('using stats.gamma.fit:')
print (stats.gamma.fit( testData, floc=0 ))

