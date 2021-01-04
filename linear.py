'''
Created on 27.11.2019

@author: mikko
'''



'''Linear model to predict humidity'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from Tools.scripts.dutree import display
from matplotlib.pyplot import plot
from sklearn.metrics import explained_variance_score

x_train = pd.read_csv('weather_data_train.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
y_train =pd.read_csv('weather_data_train_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
y_test = pd.read_csv('weather_data_test_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
x_test = pd.read_csv('weather_data_test.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)

dataset=pd.concat((x_train,y_train),axis=1)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred=regr.predict(x_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print(y_pred[:10])
#print(y_test[:10])
#print('Variance score: %.2f' %  explained_variance_score(y_test, y_pred)  )

#print(np.round(y_pred[:,0]))
regr2 = linear_model.LinearRegression()

scaler = StandardScaler()
s = scaler.fit_transform(x_train)
pca = PCA(n_components=9)
pca = pca.fit_transform(s)
#plt.figure()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Explained Variance')
#plt.show()


s2 = scaler.fit_transform(x_test)
pca1 =PCA(n_components=9)
pca1=pca1.fit_transform(s2)
regr2 = linear_model.LinearRegression()

regr2.fit(pca, y_train)

pca_pred = regr2.predict(pca1)
print('Variance score: %.2f' %  explained_variance_score(y_test, pca_pred)  )
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pca_pred)))
print(pca_pred[:10])
print(np.round(pca_pred[:10]))
print(y_test[:10])















