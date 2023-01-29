#!/usr/bin/env python
# coding: utf-8

# #Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

#ignore warnings
warnings.filterwarnings('ignore')

#seaborn visualization set up

sns.set_style('darkgrid')


# #reading dataset

# In[2]:


data=pd.read_csv('diabetes.csv')
data.head(10)


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.columns


# In[6]:


#plotting of distribution of outcome
sns.countplot(x ='Outcome',data=data)


# In[7]:


plt.figure(1)
plt.subplot(121),sns.displot(data['Glucose'])
plt.subplot(122),data['Glucose'].plot.box(figsize=(16,5))
plt.show


# In[8]:



plt.figure(2)
plt.subplot(121),sns.displot(data['Pregnancies'])
plt.subplot(122),data['Pregnancies'].plot.box(figsize=(16,5))
plt.show


# In[9]:



plt.figure(3)
plt.subplot(121),sns.displot(data['BMI'])
plt.subplot(122),data['BMI'].plot.box(figsize=(16,5))
plt.show


# In[10]:



plt.figure(4)
plt.subplot(121),sns.displot(data['Age'])
plt.subplot(122),data['Age'].plot.box(figsize=(16,5))
plt.show


# In[11]:



plt.figure(5)
plt.subplot(121),sns.displot(data['BloodPressure'])
plt.subplot(122),data['BloodPressure'].plot.box(figsize=(16,5))
plt.show


# In[12]:



plt.figure(6)
plt.subplot(121),sns.displot(data['SkinThickness'])
plt.subplot(122),data['SkinThickness'].plot.box(figsize=(16,5))
plt.show


# In[13]:



plt.figure(7)
plt.subplot(121),sns.displot(data['Insulin'])
plt.subplot(122),data['Insulin'].plot.box(figsize=(16,5))
plt.show


# In[14]:


#check missing values
data.isnull().sum()


# In[15]:


sns.pairplot(data)


# In[16]:


#check correlation
data.corr()


# In[17]:


matrix = data.corr()
ax = plt.subplots(figsize=(9,6)),sns.heatmap(matrix,vmax=.8,square=True,cmap='coolwarm')


# In[18]:


#splitting dataset
X = data.drop('Outcome',axis=1)
y = data['Outcome']


# In[19]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# XGBoost

# In[20]:


from xgboost import XGBClassifier
xgb_model=XGBClassifier(gamma=0)
xgb_model.fit(X_train,y_train)


# In[21]:


xgb_pred=xgb_model.predict(X_test)


# In[22]:


#Getting Accuracy Score for the XGBoost
from sklearn import metrics
print("Accuracy Score=",format(metrics.accuracy_score(y_test, xgb_pred)))


# In[23]:


#Metrics for XGBoost
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,xgb_pred))
print(classification_report(y_test,xgb_pred))


# In[24]:


#Getting Features Importance
xgb_model.feature_importances_


# In[25]:


#Plotting Feature Importance
(pd.Series(xgb_model.feature_importances_,index=X.columns)
 .plot(kind='barh'))


# In[26]:


#Printing prediction probability for the data
print('Prediction Probabilities',xgb_model.predict_proba(X_test))


