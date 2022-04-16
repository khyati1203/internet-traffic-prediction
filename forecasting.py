# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:13:41 2022

@author: parth
"""

from gettext import install

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
import itertools
import warnings
from math import sqrt
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
import streamlit as st




st.title('Internet Traffic Prediction')
def user_input_features():
    Years = st.number_input('Date:', min_value=0, max_value=1, value=1, step=1)
    return Years 


import pandas as pd 
from datetime import datetime





data = pd.read_csv("data.csv",
                           parse_dates=['Date'],
                           index_col='Date')


# st.write(df)

df =user_input_features()
# st.subheader('User Input parameters')
# st.write(df)

#Model Building
#future_dates=[data.index[-1]+ DateOffset(=x)for x in range(0,df)]
#future_data=pd.DataFrame(index=future_dates[1:],columns=data.columns)



#final_arima.fittedvalues.tail()

future_dates=[data.index[-1]+ DateOffset(months=x)for x in range(0,df)]
future_data=pd.DataFrame(index=future_dates[1:],columns=data.columns)
final_arima = ARIMA(data['Daily Visitors'],order = (1,1,0))
final_arima = final_arima.fit()
final_arima.fittedvalues.tail()
future_data['Daily Visitors'] = final_arima.predict(start = 120, end = 123, dynamic= True) 

future_data.astype(int)

st.subheader(f'Forecasting visitor')
st.write(future_data.astype(int))






