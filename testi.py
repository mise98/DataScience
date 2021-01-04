import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import decomposition, preprocessing
import numpy as np
x_train = pd.read_csv('weather_data_train.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
y_train =pd.read_csv('weather_data_train_labels.csv',index_col='datetime',sep=';',decimal=',',infer_datetime_format=True)
df2=pd.concat((x_train,y_train),axis=1)
x_test = pd.read_csv('weather_data_test.csv')
y_test = pd.read_csv('weather_data_test_labels.csv')
corrMatrix = x_train.corr()
#df3 = pd.concat((x_test,y_test),axis=1)

df =df2

#sns.pairplot(df2, vars= ['T_mu','P_mu','Td_mu','Ff_mu','VV_mu','U_mu'])
#sns.pairplot(df2, vars= ['VV_mu','U_mu'])
#model = LinearRegression()



#df.hist(column=["Tn_mu", "Tx_mu"])

#sns.pairplot(df2, vars= ['T_mu','P_mu','Td_mu','Ff_mu','VV_mu','U_mu'])


#plt.show()
pca_data = preprocessing.scale(df2)

pca = decomposition.PCA(n_components=2)
pca = pca.fit(pca_data)
tranformed_pca=pca.transform(pca_data)
cum_explained_var = []
for i in range(0, len(pca.explained_variance_ratio_)):
    if i == 0:
        cum_explained_var.append(pca.explained_variance_ratio_[i])
    else:
        cum_explained_var.append(pca.explained_variance_ratio_[i] + 
                                 cum_explained_var[i-1])

print(cum_explained_var)



