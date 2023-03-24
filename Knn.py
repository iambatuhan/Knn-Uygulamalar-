import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
veri=sns.load_dataset("tips")
sex=veri.iloc[:,2:3]
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()
sex=ohe.fit_transform(sex).toarray()
sonuc=pd.DataFrame(data=sex,index=range(244),columns=["Female","Male"])
smoker=veri.iloc[:,3:4]
smoker=ohe.fit_transform(smoker).toarray()
sonuc1=pd.DataFrame(data=smoker,index=range(244),columns=["No","Yes"])
day=veri.iloc[:,4:5]
day=ohe.fit_transform(day).toarray()
sonuc2=pd.DataFrame(data=day,index=range(244),columns=["Friday","Saturday","Sunday","Thursday"])
time=veri.iloc[:,5:6]
time=ohe.fit_transform(time).toarray()
sonuc3=pd.DataFrame(data=time,index=range(244),columns=["Dinner","Lunch"])
sonuc4=veri.iloc[:,0:1]
veri1=pd.concat([sonuc4,sonuc,sonuc2,sonuc3],axis=1)
from sklearn.model_selection import train_test_split
x=veri1.iloc[:,0:5].values
y=veri1.iloc[:,5:10].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_test=sc.fit_transform(y_test)
Y_train=sc.fit_transform(y_train)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
from  sklearn.metrics import confusion_matrix
import sklearn.metrics as sm
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(sm.accuracy_score(y_test, y_pred))

