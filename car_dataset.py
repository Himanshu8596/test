#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('D:\Machine Learning\Data\cars.csv', index_col=0)


# In[3]:


data


# In[4]:


import seaborn as sns


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


sns.pairplot(data, x_vars=['Cylinder','Disp','HP','Drat','Wt','Qsec','VS','AM','Gear','Carb'], y_vars='MPG', height=7, aspect=0.7, kind='reg')


# In[7]:


feature_cols = ['Cylinder','Disp','HP','Drat','Wt','Qsec','VS','AM','Gear','Carb']


# In[8]:


X = data[feature_cols]
X.head()


# In[11]:


Y=data.MPG


# In[12]:


Y.head()


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)


# In[14]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[15]:


from sklearn.linear_model import LinearRegression


# In[18]:


linreg=LinearRegression()
linreg.fit(X_train,Y_train)


# In[19]:


print(linreg.intercept_)


# In[20]:


print(linreg.coef_)


# In[21]:


list(zip(feature_cols,linreg.coef_))


# In[22]:


Y_pred = linreg.predict(X_test)
Y_pred


# In[23]:


Y_test


# In[24]:


true = [30.4, 21.4, 15.2, 30.4, 13.3, 32.4, 15.5, 15.8]
pred = [26.51096829, 20.2673039 , 19.56358615, 29.83011701, 15.10233962, 27.01378783, 17.68496216, 24.09958737]


# In[25]:


from sklearn import metrics


# In[29]:


# calculating MAE
print(metrics.mean_absolute_error(Y_test,Y_pred))


# In[30]:


#calculating MSE
print(metrics.mean_squared_error(Y_test,Y_pred))


# In[31]:


#calculating RMSE
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))


# In[32]:


#for data model accuracy:
linreg.score(X_test,Y_test)


# In[ ]:




