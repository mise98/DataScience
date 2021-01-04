'''
Created on 27.11.2019

@author: mikko
'''
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
x_train = pd.read_csv('weather_data_train.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
y_train =pd.read_csv('weather_data_train_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
df=pd.concat((x_train,y_train),axis=1)
y_test = pd.read_csv('weather_data_test_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
x_test = pd.read_csv('weather_data_test.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
'''corrMatrix = df.corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(corrMatrix, fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()'''
  

Y_train = y_train['OBSERVED']


Y_test =y_test['OBSERVED']
error = []

scaler = StandardScaler()
s = scaler.fit_transform(x_train)
pca = PCA(n_components=9)
pca = pca.fit_transform(s)

scaler2 = StandardScaler()
s2 = scaler2.fit_transform(x_test)
pca2 = PCA(n_components=9)
pca2 = pca2.fit_transform(s2)
# Calculating error for K values between 1 and 40
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(pca, Y_train.values)
    pred_i = knn.predict(pca2)
    error.append(np.mean(pred_i != Y_test.values))

m = min(error)
min_ind = error.index(m)
print(min_ind + 1)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

