# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:35:55 2022

@author: Vijaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
from tabulate import tabulate
register_matplotlib_converters()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping

def dataset(df):

    scfi=pd.read_csv(df,parse_dates = ['date'], index_col = ['date'])
    
    #Creating Train set
    scfi_train_w    = scfi[0:620]
    #Creating Test set
    scfi_test_w     = scfi[620:627]
    scfi_train_w.describe()
    scfi_train_w.skew()
    scfi_train_w.kurt()
    #creating log for traina nd test
    scfi_train_w_log   = np.log(scfi_train_w)
    scfi_test_w_log     =np.log(scfi_test_w) 

    return scfi,scfi_train_w,scfi_test_w,scfi_train_w_log,scfi_test_w_log

def arima_7days(scfi_train_w_log,scfi_test_w):
    scfi_train_w =scfi_train_w_log.reset_index()
    y_scfi_w = scfi_train_w['value']
    x_scfi_w = range(0, 620)
    scfi_train_w.columns.values
    scfi_w = smf.ols('y_scfi_w ~ x_scfi_w', data =scfi_train_w)
    scfi_w=  scfi_w.fit()
    scfi_w.summary()
    #plt.plot(gemval_w.fittedvalues)
    #plt.plot(y_gemval_w)

    scfi_stationary_w = y_scfi_w - scfi_w.fittedvalues
    #plt.plot(gemval_stationary_w)
    #plot_acf(gemval_stationary_w)
    #plot_pacf(gemval_stationary_w) #AR1 should be good, although number 14 is a bit worring.
    scfi_ar1_w = ARIMA(y_scfi_w, order = (1, 0, 0))
    scfi_ar1_w = scfi_ar1_w.fit()
    #fig = plt.figure() 
    #ax1 = fig.add_subplot(3, 1, 1) # number of row and column + position
    #ax2 = fig.add_subplot(3, 1, 2)
    #ax3 = fig.add_subplot(3, 1, 3)
    #ax1.plot(gemval_ar1_w.resid)
    #plot_acf(gemval_ar1_w.resid, ax = ax2)
    #plot_pacf(gemval_ar1_w.resid, ax = ax3)
    #fig.tight_layout()
    #plt.show() #all the short term correlation and random variation seem to be removed

    scfi_ar1_pred_w = scfi_ar1_w.predict(start = 620, end = 627)
    #plt.plot(gemval_ar1_pred_w)
    # make the predictions for 1

    scfi_ar1_2_w = SARIMAX(y_scfi_w, order = (1, 0, 0))
    scfi_ar1_2_w = scfi_ar1_2_w.fit()
    scfi_ar1_2_w.summary() 


    y_pred_scfi_w = scfi_ar1_2_w.get_forecast(len(scfi_test_w.index)+1)
    y_pred_df_scfi_w = y_pred_scfi_w.conf_int(alpha = 0.05) 
    y_pred_df_scfi_w["Predictions"] = scfi_ar1_2_w.predict(start = y_pred_df_scfi_w.index[0], end = y_pred_df_scfi_w.index[-1])
    #conf_df = pd.concat([test['MI'],predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
   
    #conf_df.head()
    #fig = plt.figure(figsize = (16,8))
    #ax1 = fig.add_subplot(1, 1, 1)
    #plt.plot(y_gemval_w)
    #plt.plot(y_pred_df_gemval_w["Predictions"],label='predicted')
    y_pred_df_scfi_w.head()
    #plt.plot(y_pred_df_gemval_w['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
    #plt.plot(y_pred_df_gemval_w['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
    #plt.fill_between(y_pred_df_gemval_w["Predictions"].index.values,y_pred_df_gemval_w['upper value'],                     color = 'grey', alpha = 0.2)
    #plt.legend(loc = 'lower left', fontsize = 12)
    #plt.show()
    scfi_ar_pred_w = pd.DataFrame(scfi_ar1_pred_w)
    scfi_test_w['pred_1'] = scfi_ar_pred_w.values[1:8]
    rmspe1_scfi_1_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 1])**2)/7)
    
    
    y_pred_df_scfi_w.columns.values
    pred_2_w = pd.DataFrame(y_pred_df_scfi_w['Predictions'])
    scfi_test_w['pred_2_w'] = pred_2_w.values[1:8]
    rmspe1_scfi_2_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 1])**2)/7)
    #ARIMA model (1 0 1)
    scfi_ar1_w2 = ARIMA(y_scfi_w, order = (1, 0, 1))
    scfi_ar1_w2 = scfi_ar1_w2.fit()


                                                                         
   
    scfi_ar1_pred_w2 = scfi_ar1_w2.predict(start = 620, end = 627)
    #plt.plot(y_scfi_w)
    #plt.plot(scfi_ar1_pred_w2)
    

    scfi_ar1_2_w2 = SARIMAX(y_scfi_w, order = (1, 1, 0)) #(1 0 1) des not become stationary
    scfi_ar1_2_w2 = scfi_ar1_2_w2.fit()
    y_pred_scfi_w2 = scfi_ar1_2_w2.get_forecast(len(scfi_test_w.index)+1)
    y_pred_df_scfi_w2 = y_pred_scfi_w2.conf_int(alpha = 0.05) 
    y_pred_df_scfi_w2["Predictions"] = scfi_ar1_2_w2.predict(start = y_pred_df_scfi_w2.index[0], end = y_pred_df_scfi_w2.index[-1])

    scfi_ar_pred_w2 = pd.DataFrame(scfi_ar1_pred_w2)
    scfi_test_w['pred_1'] = scfi_ar_pred_w2.values[1:8]
    rmspe2_scfi_1_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 1])**2)/7)

    y_pred_df_scfi_w2.columns.values
    pred_2_w2 = pd.DataFrame(y_pred_df_scfi_w2['Predictions'])
    scfi_test_w['pred_2_w'] = pred_2_w2.values[1:8]
    rmspe2_scfi_2_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 2])**2)/7)
    return y_scfi_w,y_pred_df_scfi_w2,scfi_ar1_2_w.summary()


def arima_30days(df):
    #ARIMA 30 DAYS
    scfi_Train_m    = df[0:597]
    scfi_Test_m     = df[597:627]
    scfi_Train_m    = np.log(scfi_Train_m)
    scfi_Test_m    = np.log(scfi_Test_m)
    

    scfi_Train_m = scfi_Train_m.reset_index()
    y_scfi_m = scfi_Train_m['value']
    x_scfi_m = range(0, 597)
    scfi_lm_m = smf.ols('y_scfi_m ~ x_scfi_m', data = scfi_Train_m)
    scfi_lm_m = scfi_lm_m.fit()
    scfi_lm_m.summary()
   
    scfi_stationary_m = y_scfi_m - scfi_lm_m.fittedvalues
    

    scfi_ar_m = ARIMA(y_scfi_m, order = (1, 0, 0))
    scfi_ar_m = scfi_ar_m.fit()

    

    scfi_ar_pred_m = scfi_ar_m.predict(start = 597, end = 627)

   
    scfi_ar_2_m = SARIMAX(y_scfi_m, order = (1, 0, 0))

    scfi_ar_2_m = scfi_ar_2_m.fit()
    y_pred_scfi_m = scfi_ar_2_m.get_forecast(len(scfi_Test_m.index)+1)
    y_pred_df_scfi_m = y_pred_scfi_m.conf_int(alpha = 0.05) 
    y_pred_df_scfi_m["Predictions"] = scfi_ar_2_m.predict(start = y_pred_df_scfi_m.index[0], end = y_pred_df_scfi_m.index[-1])
    
    scfi_ar_pred_m = pd.DataFrame(scfi_ar_pred_m)
    scfi_Test_m['pred_1'] = scfi_ar_pred_m.values[1:32]
    rmspe_scfi_1_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 1])**2)/30)

    pred_2_m = pd.DataFrame(y_pred_df_scfi_m['Predictions'])
    scfi_Test_m['pred_2_m'] = pred_2_m.values[1:32]
    rmspe__scfi_2_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 2])**2)/30)

    #Arima(1 0 1)

    scfi_ar1_m2 = ARIMA(y_scfi_m, order = (1, 0, 1))
    scfi_ar1_m2 = scfi_ar1_m2.fit()

     

    scfi_ar1_pred_m2 = scfi_ar1_m2.predict(start = 597, end = 627)
    

    scfi_ar1_2_m2 = SARIMAX(y_scfi_m, order = (1, 1, 0)) #(1 0 1) des not become stationary
    scfi_ar1_2_m2 = scfi_ar1_2_m2.fit()
    y_pred_scfi_m2 = scfi_ar1_2_m2.get_forecast(len(scfi_Test_m.index)+1)
    y_pred_df_scfi_m2 = y_pred_scfi_m2.conf_int(alpha = 0.05) 
    y_pred_df_scfi_m2["Predictions"] = scfi_ar1_2_m2.predict(start = y_pred_df_scfi_m2.index[0], end = y_pred_df_scfi_m2.index[-1])
    
    scfi_ar_pred_m2 = pd.DataFrame(scfi_ar1_pred_m2)
    scfi_Test_m['pred_1'] = scfi_ar_pred_m2.values[1:32]
    rmspe2_scfi_1_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 1])**2)/29)

    y_pred_df_scfi_m2.columns.values
    pred_2_m2 = pd.DataFrame(y_pred_df_scfi_m2['Predictions'])
    scfi_Test_m['pred_2_m'] = pred_2_m2.values[1:32]
    rmspe2_scfi_2_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 2])**2)/29)
    return y_scfi_m,y_pred_df_scfi_m2,scfi_ar1_m2.summary()


def arima_180days(df):
    scfi_Train_6m                = df[0:447]
    scfi_Test_6m                = df[447:627]
    scfi_Train_6m                = np.log(scfi_Train_6m)
    scfi_Test_6m                = np.log(scfi_Test_6m)

    scfi_Train_6m = scfi_Train_6m.reset_index()
    y_scfi_6m = scfi_Train_6m['value']
    x_scfi_6m = range(0, 447)
    scfi_Train_6m.columns.values
    scfi_lm_6m = smf.ols('y_scfi_6m ~ x_scfi_6m', data = scfi_Train_6m)
    scfi_lm_6m = scfi_lm_6m.fit()
    scfi_lm_6m.summary()
    
    scfi_stationary_6m = y_scfi_6m - scfi_lm_6m.fittedvalues
    #plot_pacf(gemval_stationary_6m)# AR1 ok, higher numbers problematics

    scfi_ar_6m = ARIMA(y_scfi_6m, order = (1, 0, 0))
    scfi_ar_6m = scfi_ar_6m.fit()



   

    scfi_ar_pred_6m2 = scfi_ar_6m.predict(start = 447, end = 627)

    
    scfi_ar_2_6m2 = SARIMAX(y_scfi_6m, order = (2, 0, 0))
    scfi_ar_2_6m2 = scfi_ar_2_6m2.fit()
    y_pred_scfi_6m2 = scfi_ar_2_6m2.get_forecast(len(scfi_Test_6m.index)+1)
    y_pred_df_scfi_6m2 = y_pred_scfi_6m2.conf_int(alpha = 0.05) 
    y_pred_df_scfi_6m2["Predictions"] = scfi_ar_2_6m2.predict(start = y_pred_df_scfi_6m2.index[0], end = y_pred_df_scfi_6m2.index[-1])
        


    scfi_ar_pred6_m2 = pd.DataFrame(scfi_ar_pred_6m2)
    scfi_Test_6m['pred_1'] = scfi_ar_pred_6m2.values[1:181]
    rmspe__scfi_1_6m2 = np.sqrt(np.sum(scfi_Test_6m.iloc[:, 0].subtract(scfi_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_scfi_6m2.columns.values
    pred_2_6m2 = pd.DataFrame(y_pred_df_scfi_6m2['Predictions'])
    scfi_Test_6m['pred_2_6m'] = pred_2_6m2.values[1:181]
    rmspe_scfi_2_6m2 = np.sqrt(np.sum(scfi_Test_6m.iloc[:, 0].subtract(scfi_Test_6m.iloc[:, 2])**2)/180)
        
    return y_scfi_6m,y_pred_df_scfi_6m2,scfi_ar_6m.summary()
def LSTM_7days(df):
    scfi_train_w    = df[0:620]
    #Creating Test set
    scfi_test_w     = df[620:627]
    
    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(scfi_train_w['value'].values.reshape(-1,1))
    # how many days do i want to base my predictions on ?
    prediction_days = 7 

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    
    model = Sequential()
    
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
  

    model.summary()
    model.compile(optimizer='adam', 
                  loss='mean_squared_error')
    model.fit(x_train, 
          y_train, 
          epochs=50, 
          batch_size = 32,
         )

    # test model accuracy on existing data
    test_data = scfi_test_w 

    actual_prices = test_data.values

    total_dataset = pd.concat((scfi_train_w, test_data), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    mae = mean_absolute_error(test_data['value'],  predicted_prices)

    rmspe_gemval_lstm_w = np.sqrt(mean_squared_error(test_data['value'],  predicted_prices))
    e= ['mae', 'rmspe_gemval_lstm_w']
    eval = pd.DataFrame([mae, rmspe_gemval_lstm_w], index=e, columns=['Score'])
    
    return actual_prices,predicted_prices,eval

def LSTM_30days(df):
    scfi_train_w    = df[0:620]
    #Creating Test set
    scfi_test_w     = df[620:627]
    scfi_train_w   = np.log(scfi_train_w)
    scfi_test_w     =np.log(scfi_test_w) 
    y_test=scfi_test_w['value'].values

    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_scfi_w =  scaler.fit_transform(scfi_train_w)
    X_train = []
    y_train = []
    for i in range(30, len(scfi_train_w)-7):
        X_train.append(lstm_scfi_w[i-7:i, 0])
        y_train.append(lstm_scfi_w[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
     

    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)


    dataset_train_scfi_w = scfi_train_w.iloc[:613]
    dataset_test_scfi_w = scfi_train_w.iloc[613:]
    dataset_total_scfi_w = pd.concat((dataset_train_scfi_w, dataset_test_scfi_w), axis = 0)
    inputs_scfi_w = dataset_total_scfi_w[len(dataset_total_scfi_w) - len(dataset_test_scfi_w) - 7:].values
    inputs_scfi_w = inputs_scfi_w.reshape(-1,1)
    inputs_scfi_w = scaler.transform(inputs_scfi_w)
    X_test_scfi_w = []
    for i in range(7, 14):
        
        X_test_scfi_w.append(inputs_scfi_w[i-7:i, 0])
    X_test_scfi_w = np.array(X_test_scfi_w)
    X_test_scfi_w = np.reshape(X_test_scfi_w, (X_test_scfi_w.shape[0], X_test_scfi_w.shape[1], 1))
   
    pred_scfi_w = model.predict(X_test_scfi_w)
    pred_scfi_w = scaler.inverse_transform(pred_scfi_w)
    
    mae = mean_absolute_error(scfi_test_w['value'], pred_scfi_w)

    rmspe_gemval_lstm_w = np.sqrt(mean_squared_error(scfi_test_w['value'], pred_scfi_w))
    e= ['mae', 'rmspe_scfi_lstm_w']
    eval = pd.DataFrame([mae, rmspe_scfi_lstm_w], index=e, columns=['Score'])
   
    return y_test,pred_scfi_w,eval
    
def LSTM_30days(df):
    scfi_train_m    = df[0:597]
    #Creating Test set
    scfi_test_m     = df[597:627]
    scfi_train_m   = np.log(scfi_train_m)
    scfi_test_m     =np.log(scfi_test_m) 
    y_test=scfi_test_m['value'].values
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_scfi_m =  scaler.fit_transform(scfi_train_m)
    X_train = []
    y_train = []
    for i in range(30, len(scfi_train_m)-30):
        X_train.append(lstm_scfi_m[i-30:i, 0])
        y_train.append(lstm_scfi_m[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    dataset_train_scfi_m = scfi_train_m.iloc[:567]
    dataset_test_scfi_m = scfi_train_m.iloc[567:]
    y_test1=dataset_test_scfi_m['value'].values
    dataset_total_scfi_w = pd.concat((dataset_train_scfi_m, dataset_test_scfi_m), axis = 0)
    inputs_scfi_m = dataset_total_scfi_w[len(dataset_total_scfi_w) - len(dataset_test_scfi_m) - 30:].values
    inputs_scfi_m = inputs_scfi_m.reshape(-1,1)
    inputs_scfi_m = scaler.transform(inputs_scfi_m)
    X_test_scfi_m = []
    for i in range(30, 60):
    
        X_test_scfi_m.append(inputs_scfi_m[i-30:i, 0])
    X_test_scfi_m = np.array(X_test_scfi_m)
    X_test_scfi_m = np.reshape(X_test_scfi_m, (X_test_scfi_m.shape[0], X_test_scfi_m.shape[1], 1))
    pred_scfi_m = model.predict(X_test_scfi_m)
    pred_scfi_m = scaler.inverse_transform(pred_scfi_m)
    mae = mean_absolute_error(scfi_test_m['value'], pred_scfi_m)
    
    rmspe_scfi_lstm_m = np.sqrt(mean_squared_error(scfi_test_m['value'], pred_scfi_m))
    
    e= ['mae', 'rmspe_scfi_lstm_m']
    eval = pd.DataFrame([mae, rmspe_scfi_lstm_m], index=e, columns=['Score'])
   
    return y_test,pred_scfi_m,eval
    
def LSTM_180days(df):    
    scfi_train_6m    = df[0:447]
    #Creating Test set
    scfi_test_6m     = df[447:627]
    scfi_train_6m    = np.log(scfi_train_6m) 
    scfi_test_6m    = np.log( scfi_test_6m)
    y_test=scfi_test_6m['value'].values    
    
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_scfi_6m =  scaler.fit_transform(scfi_train_6m)
    X_train = []
    y_train = []
    for i in range(180, len(scfi_train_6m)-180):
        X_train.append(lstm_scfi_6m[i-180:i, 0])
        y_train.append(lstm_scfi_6m[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    dataset_train_scfi_6m = scfi_train_6m.iloc[:267]
    dataset_test_scfi_6m = scfi_train_6m.iloc[267:]
    y_test1=dataset_test_scfi_6m['value'].values
    dataset_total_scfi_6m = pd.concat((dataset_train_scfi_6m, dataset_test_scfi_6m), axis = 0)
    inputs_scfi_6m = dataset_total_scfi_6m[len(dataset_total_scfi_6m) - len(dataset_test_scfi_6m) - 180:].values
    inputs_scfi_6m = inputs_scfi_6m.reshape(-1,1)
    inputs_scfi_6m = scaler.transform(inputs_scfi_6m)
    X_test_scfi_6m = []
    for i in range(180, 360):
        X_test_scfi_6m.append(inputs_scfi_6m[i-180:i, 0])
    X_test_scfi_6m = np.array(X_test_scfi_6m)
    X_test_scfi_6m = np.reshape(X_test_scfi_6m, (X_test_scfi_6m.shape[0], X_test_scfi_6m.shape[1], 1))
    pred_scfi_6m = model.predict(X_test_scfi_6m)
    pred_scfi_6m = scaler.inverse_transform(pred_scfi_6m)
    mae = mean_absolute_error(scfi_test_6m['value'], pred_scfi_6m)
    rmspe_scfi_lstm_6m = np.sqrt(mean_squared_error(scfi_test_6m['value'], pred_scfi_6m))
    
    e= ['mae', 'rmspe_scfi_lstm_6m']
    eval = pd.DataFrame([mae, rmspe_scfi_lstm_6m], index=e, columns=['Score'])
   
    return y_test,pred_scfi_6m,eval