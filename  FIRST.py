#!/usr/bin/env python
# coding: utf-8

# In[3]:


#credit


# In[4]:


import pandas as pd


# In[13]:


data = pd.read_excel("Upgrade.xlsx")


# In[14]:


#Train-Test Split 


# In[17]:


import statsmodels.formula.api as sm 


# In[18]:


import statsmodels.api as sma


# In[19]:


# glm stands for generalized linear model


# In[20]:


mylogit = sm.glm( formula = "upgraded ~ purchases + extraCards", data = data, family = sma.families.Binomial()).fit()


# In[21]:


mylogit.summary()


# In[40]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
import itertools
import subprocess 
from time import time
from scipy import stats
import scipy.optimize as opt
from scipy.stats import chi2_contingency
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve


# In[38]:


credittrain, credittest = train_test_split(data, train_size=0.70, random_state=1)


# In[39]:


#traintestspli ^


# In[28]:


import statsmodels.api as sm 


# In[29]:


#defining the dpenedent and the independent variables 


# In[41]:


Xtrain = credittrain[['purchases', 'extraCards']]


# In[43]:


ytrain = credittrain[['upgraded']]


# In[32]:


# building the model and fitting the data


# In[44]:


log_reg = sm.Logit(ytrain, Xtrain).fit()


# In[45]:


print(log_reg.summary())


# In[46]:


from sklearn.metrics import (confusion_matrix, accuracy_score)


# In[47]:


#confusion matrix


# In[57]:


Xtest = credittest[['purchases', 'extraCards']]
ytest = credittest['upgraded']


# In[58]:


# preforming predictions on the test dataset


# In[59]:


yhat = log_reg.predict(Xtest)
prediction = list(map(round, yhat))


# In[61]:


# comparing original and predicted values of y 

print('Actual values', list(ytest.values))
print('Predictions :', prediction)


# In[62]:


cm = confusion_matrix(ytest, prediction)
print ('Confusion Matrix : \n', cm)


# In[63]:


# accuracy score of the model
print('Test accuracy =', accuracy_score(ytest, prediction))


# In[68]:




