'''
Created on 28.11.2019

@author: mikko
'''

'''Makes knn classification and plots confusion matrix'''

import numpy as np 
import seaborn as sn
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt  
import seaborn as sns 
from linear import y_pred
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import difflib

x_train = pd.read_csv('weather_data_train.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
y_train =pd.read_csv('weather_data_train_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
y_test = pd.read_csv('weather_data_test_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
x_test = pd.read_csv('weather_data_test.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
df_train = y_train['OBSERVED']

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(x_train, df_train.values)

y_pred=knn.predict(x_test)

print(y_pred[:30])

df_test=y_test['OBSERVED']

print(df_test.values[:30])

print(confusion_matrix( df_test.values, y_pred))
'''df_cm = pd.DataFrame(confusion_matrix( df_test.values, y_pred), range(2),range(2))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})'''

components=9
scaler = StandardScaler()
s = scaler.fit_transform(x_train)
pca = PCA(n_components=components)
pca = pca.fit_transform(s)

scaler2 = StandardScaler()
s2 = scaler2.fit_transform(x_test)
pca2 = PCA(n_components=components)
pca2 = pca2.fit_transform(s2)
knn2 = KNeighborsClassifier(n_neighbors=39)

knn2.fit(pca, df_train.values)
print(df_test.values)
pca_pred =knn2.predict(pca2)
sm = difflib.SequenceMatcher(None,df_test.values,y_pred)
print(sm)
print(pca_pred[:20])

df_cm2 = pd.DataFrame(confusion_matrix( df_test.values, pca_pred), range(2),range(2))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm2, annot=True,annot_kws={"size": 16})# font size
print(df_cm2)
plt.show()

















