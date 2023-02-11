from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential # pip install keras
from keras.layers import Dense # pip install tensorflow (as administrator)
import os, math

df = pd.read_csv("serieFcast2021.csv")
ds = df[df.columns[1]] # converts to series
plt.plot(df.amount,label="Fcast")
plt.grid()
plt.legend()
plt.show()


#Preprocessing: adding missing values
ds1=ds.interpolate()
plt.plot(ds1,label="Fcast")
plt.grid()
plt.show()

x = df.iloc[:, 0]
y = ds1

#################################

#Seasonal decompose
result1 = seasonal_decompose(ds1, model='multiplicative',period=12)
result1.plot()
plt.show()

npa = ds1.to_numpy()
logdata = np.log(npa)
plt.plot(npa, color = 'blue', marker = "o")
plt.plot(logdata, color = 'red', marker = "o")
plt.title("numpy.log()")
plt.xlabel("x");plt.ylabel("logdata")
plt.show()

#ACF
from statsmodels.tsa.stattools import acf
diffdata = ds1.diff()
diffdata[0] = ds1[0] # reset 1st elem
acfdata = acf(diffdata,unbiased=True,nlags=24)
plt.bar(np.arange(len(acfdata)),acfdata)
plt.show()
# otherwise
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(diffdata, lags=24)
plt.show()

#Logdiff
logdiff = pd.Series(logdata).diff()
cutpoint = int(0.7*len(logdiff))
trainLog = logdiff[:cutpoint]
testLog = logdiff[cutpoint:]
trainLog[0] = 0 # set first entry
reconstruct = np.exp(np.r_[trainLog,testLog].cumsum()+logdata[0])


##################################
#Neural network mlp method

def create_dataset(dataset, look_back=1):   
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


np.random.seed(550) # for reproducibility

df = pd.read_csv("serieFcast2021.csv", usecols=[1], names=['amount'],header=0).interpolate()

dataset = df.values # time series values
dataset = dataset.astype('float32') # needed for MLP input
# split into train and test sets
train_size = int(len(dataset) -12)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print("Len train={0}, len test={1}".format(len(train), len(test)))
# sliding window matrices (look_back = window width); dim = n - look_back - 1
look_back = 2
testdata = np.concatenate((train[-look_back:],test))
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(testdata, look_back)
# Multilayer Perceptron model
loss_function = 'mean_squared_error'
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu')) # 8 hidden neurons
model.add(Dense(1)) # 1 output neuron
model.compile(loss=loss_function, optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore,
math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))
# generate predictions for training and forecast for plotting
trainPredict = model.predict(trainX)
testForecast = model.predict(testX)

# predict for 24 more periods
result = testForecast
for n in np.linspace(0, 23, 24):
    np.append(result,model.predict(np.asarray([[result[-2:,0][0],result[-1:,0][0]]],
                                              dtype="float32")))

plt.plot(dataset)
plt.plot(np.concatenate((np.full(look_back-1, np.nan), trainPredict[:,0])))
plt.plot(np.concatenate((np.full(len(train)-1, np.nan), testForecast[:, 0])))
plt.plot(np.concatenate((np.full(len(train)+11, np.nan), result[:, 0])))
plt.show()

print("MSE={}".format(mean_absolute_error(testY, testForecast)))


######################################
#Machine learning method

mdl = rf = RandomForestRegressor(n_estimators=500)
mdl.fit(trainX, trainY)
trainPredict = mdl.predict(trainX)
testForecast = mdl.predict(testX)

result = testForecast
for n in np.linspace(0, 23, 24):
    np.append(result,model.predict(np.asarray([[result[-2:-1],result[-1:]]],
                                              dtype="float32")))

plt.plot(dataset)
plt.plot(np.concatenate((np.full(look_back-1, np.nan), trainPredict)))
plt.plot(np.concatenate((np.full(len(train)-1, np.nan),testForecast)))
plt.plot(np.concatenate((np.full(len(train)+11, np.nan), result)))
plt.show()
print("MSE={}".format(mean_absolute_error(testY, testForecast)))

######################################


#Statistical prediction method
#Non preprocessed data
#Forecast in sample and out
sarima_model = SARIMAX(ds1, order=(1,0,0), seasonal_order=(0,1,1,12))
sfit = sarima_model.fit()
sfit.plot_diagnostics(figsize=(10, 6))
plt.show()
ypred = sfit.predict(start=0,end=len(ds1)-1)
print("MSE={}".format(mean_absolute_error(ds1,ypred)))

forewrap = sfit.get_forecast(steps=24)
forecast_ci = forewrap.conf_int()
forecast_val = forewrap.predicted_mean
plt.plot(ds1)
plt.plot(ypred)
plt.plot(np.linspace(len(ds1),len(ds1)+24,24),forecast_val)
plt.xlabel('time');plt.ylabel('amount')
plt.show()


#Preprocessed data
#Forecasting in sample and out
sarima_model = SARIMAX(trainLog, order=(1,0,0), seasonal_order=(0,1,1,12))
sfit = sarima_model.fit()
sfit.plot_diagnostics(figsize=(10, 6))
plt.show()
ypred = sfit.predict(start=0,end=len(ds1)-1)
print("MSE={}".format(mean_absolute_error(testLog,ypred[cutpoint:])))

forewrap = sfit.get_forecast(steps=24)
forecast_ci = forewrap.conf_int()
forecast_val = forewrap.predicted_mean
plt.plot(logdiff)
plt.plot(ypred)
plt.plot(np.linspace(len(logdiff),len(logdiff)+24,24),forecast_val)
plt.xlabel('time');plt.ylabel('amount')
plt.show()


#Sarima parameter fitting and forecasting out of sample for 24 periods
model = pm.auto_arima(ds1, start_p=1, start_q=1,
test='adf', max_p=3, max_q=3, m=12,
start_P=0, seasonal=True,
d=None, D=1, trace=True,
error_action='ignore',
suppress_warnings=True,
stepwise=True) # False full grid

print(model.summary())
morder = model.order
mseasorder = model.seasonal_order
fitted = model.fit(ds1)
yfore = fitted.predict(n_periods=24) # forecast
ypred = fitted.predict_in_sample()
plt.plot(ds1.values)
plt.plot(ypred)
plt.plot([None for i in ypred] + [x for x in yfore])
plt.xlabel('time');plt.ylabel('amount')
plt.show()


