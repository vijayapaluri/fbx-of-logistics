import streamlit as st
import pandas as pd

import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
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


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import scfi
st.markdown("<h1 style='text-align: center; color:#F08080 ;'>China-Shanghai Freight Index Logistics</h1>", unsafe_allow_html=True)
Models=st.sidebar.selectbox("Models", ["ARIMA","LSTM"])
days=st.sidebar.selectbox("days", ["7 Days","30 Days","180 Days"])
Refresh=st.sidebar.button("Refresh")

if Refresh:    
     st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Dataset</h3>", unsafe_allow_html=True)
     df,train,test,train_log,test_log=scfi.dataset("https://raw.githubusercontent.com/vijayapaluri/fbx-of-logistics/main/freight_index.csv")
     st.write(df) 
     st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Summary Statistics</h3>", unsafe_allow_html=True) 
     st.dataframe(df.describe())
     st.dataframe(df.skew())
     st.dataframe(df.kurt())
     st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Before Predictions on Train values</h5>", unsafe_allow_html=True)
     fig=plt.figure(figsize=(12,6))
     ax1 = fig.add_subplot(1, 1, 1)
     ax1.set_facecolor('#EAF2F8')
     plt.plot(df)
     plt.xlabel("date")
     plt.ylabel("value")
     plt.legend(['actual','values'])
     st.pyplot(fig)
     
     if Models=="ARIMA":
         if days=="7 Days":
             y_scfi_w,y_pred_df_scfi_w,y_fit=scfi.arima_7days(train_log,test_log)
             st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 7 days</h5>", unsafe_allow_html=True)
             fig = plt.figure(figsize = (16,8))
             ax1 = fig.add_subplot(1, 1, 1)
             ax1.set_facecolor('#EAF2F8')
             plt.title("Confidence Interval after 7 days")
             plt.plot(y_scfi_w)
             plt.plot(y_pred_df_scfi_w["Predictions"],label='predicted')
             
             plt.plot(y_pred_df_scfi_w['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
             plt.plot(y_pred_df_scfi_w['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
             plt.fill_between(y_pred_df_scfi_w["Predictions"].index.values,
                              y_pred_df_scfi_w['upper value'], 
                              color = 'grey', alpha = 0.2)
             plt.legend(loc = 'lower left', fontsize = 12)
             st.pyplot(fig)
             st.write(y_fit)
         elif days=="30 Days":
             y_scfi_m,y_pred_df_scfi_m,y_fit=scfi.arima_30days(df)
             
             st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 30 days</h5>", unsafe_allow_html=True)
             fig = plt.figure(figsize = (16,8))
             ax1 = fig.add_subplot(1, 1, 1)
             ax1.set_facecolor('#EAF2F8')
             plt.title("Confidence Interval after 30 days")
             plt.plot(y_scfi_m)
             plt.plot(y_pred_df_scfi_m["Predictions"],label='predicted')
             plt.plot(y_pred_df_scfi_m['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
             plt.plot(y_pred_df_scfi_m['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
             plt.fill_between(y_pred_df_scfi_m["Predictions"].index.values,
                              y_pred_df_scfi_m['upper value'], 
                              color = 'grey', alpha = 0.2)
             plt.legend(loc = 'lower left', fontsize = 12)
             st.pyplot(fig)
             st.write(y_fit)
         elif days=="180 Days":
             y_scfi_6m,y_pred_df_scfi_6m2,scfi_ar_6m=scfi.arima_180days(df)
             st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
             fig = plt.figure(figsize = (16,8))
             ax1 = fig.add_subplot(1, 1, 1)
             ax1.set_facecolor('#EAF2F8')
             plt.title("Confidence Interval after 180 days")
             plt.plot(y_scfi_6m)
             plt.plot(y_pred_df_scfi_6m2["Predictions"],label='predicted')
             plt.plot(y_pred_df_scfi_6m2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
             plt.plot(y_pred_df_scfi_6m2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
             plt.fill_between(y_pred_df_scfi_6m2["Predictions"].index.values,
                              y_pred_df_scfi_6m2['upper value'], 
                              color = 'grey', alpha = 0.2)
             plt.legend(loc = 'lower left', fontsize = 12)
             st.pyplot(fig)
             st.write(scfi_ar_6m)
     elif Models=="LSTM": 
        if days=="7 Days":
            actual_prices,predicted_prices,eval=scfi.LSTM_7days(df)
            st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
            fig = plt.figure(figsize = (16,8))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_facecolor('#EAF2F8')
            
            plt.plot(actual_prices,label='actual')
            plt.plot(predicted_prices,label='predicted')
            plt.xlabel("date")
            plt.ylabel("value")
            plt.legend()
            st.pyplot(fig)
            st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
            st.write(eval)
        elif days=="30 Days":
            actual_prices,predicted_prices,eval=scfi.LSTM_30days(df)
            st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
            fig = plt.figure(figsize = (16,8))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_facecolor('#EAF2F8')
            
            plt.plot(actual_prices,label='actual')
            plt.plot(predicted_prices,label='predicted')
            plt.xlabel("date")
            plt.ylabel("value")
            plt.legend()
            st.pyplot(fig)
            st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
            st.write(eval)
        elif days=="180 Days":
            actual_prices,predicted_prices,eval=scfi.LSTM_180days(df)
            st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
            fig = plt.figure(figsize = (16,8))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_facecolor('#EAF2F8')
            
            plt.plot(actual_prices,label='actual')
            plt.plot(predicted_prices,label='predicted')
            plt.xlabel("date")
            plt.ylabel("value")
            plt.legend()
            st.pyplot(fig)
            st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
            st.write(eval)
            
        
        
            
         

            

