import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('breastcancer.csv')
data.head()
data.shape
data.info()
data.isnull()
data.isnull().sum()
data=data.drop('Unnamed: 32',axis=1)
data.columns
data['diagnosis'].unique()
data['diagnosis'].value_counts()
a=list(data.columns)
print(a)
data.describe()
sns.countplot(data['diagnosis'],data=None)
#Heatmap of Correlation
corr = data.corr()
plt.figure (figsize= (10,10))
sns.heatmap (corr);
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data['diagnosis'].head()
#Splitting the data into the Training and Testing set
x = data.drop ('diagnosis',axis=1)
y = data ['diagnosis']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.3)
x_train.shape
x_test.shape
#Feature Scaling of data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform (x_train)
x_test = ss.fit_transform (x_test)

#Support Vector Machine
from sklearn import svm
svc = svm.SVC ()
#Loading the training data in the model
svc.fit (x_train,y_train)
#Predicting output with test data
y_pred = svc.predict (x_test)
y_pred
#Accuracy Score of Support vector classifier
from sklearn.metrics import accuracy_score
print('Accuracy Score of Support vector classifier:')
print (accuracy_score (y_test,y_pred ))

#Accuracy Score of Support vector classifier:  0.9941520467836257
