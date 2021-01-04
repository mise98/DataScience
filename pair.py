'''
Created on 27.11.2019

@author: mikko
'''

'''Makes a pair plot of weather data set'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

x_train = pd.read_csv('weather_data_train.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
y_train =pd.read_csv('weather_data_train_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)

df2=pd.concat((x_train,y_train),axis=1)
sns.pairplot(df2, vars= ['T_mu','P_mu','Td_mu','Ff_mu','VV_mu','U_mu','OBSERVED'])
plt.show()

