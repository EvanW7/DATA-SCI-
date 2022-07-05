#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# ## Linear Regression
# Analysing housing data in Sacramento, CA, and predicting house prices.

# ### 1) Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


# ### 2) Import and analyse the housing data

# #### Performing EDA (Exploratory Data Analysis) and having fun!

# In[2]:


# Read CSV
data = pd.read_csv("data.csv")


# Getting the dimensions of the array.

# In[3]:


# print the shape of the array

## write your code and run the cell


# ##### DO NOT CHANGE, JUST RUN THE CELL
# data.info()

# Making sure if we imported the right dataset by looking at the first five entries.

# In[4]:


# Head

## print the data head

## write your code and run the cell


# Making sure if we imported the right dataset by looking at the last five entries.

# In[5]:


# Tail

## print the data tail

## write your code and run the cell


# Making sure that there are no null values.

# In[6]:


# DO NOT CHANGE, JUST RUN THE CELL
# Check Null Values
data.isnull().sum()


# Measures of change and central tendency.

# In[7]:


# DO NOT CHANGE, JUST RUN THE CELL
data.describe().T


# Perfect! everything looks good.

# ### 3) Predict Price

# Our goal is to predict the price given new information about a house in the area covered by the data.

# #### List all possible variables which might be a Predictor variable.

# In[8]:


# ANSWER

## COMMENT YOUR ANSWER HERE (CHOOSE ATLEAST 3 DIFFERENT PREDICTOR VARIABLES)

# BED, SQ FT, baths


# #### Setting the target variable

# In[9]:


# DO NOT CHANGE, JUST RUN THE CELL
def draw_scatter_plot(X, Y):
    fig = plt.figure(figsize=(15,7))
    ax = plt.gca()
    ax.scatter(X, Y, c='b')
    plt.show();


# In[10]:


# Target Variable - SET THE TARGET VARIABLE TO PRICE VALUES FROM 'data'
Y = data['price']


# In[61]:


# Predictor Variable - SET THE PREDICTOR VARIBLE TO ONE OF YOUR CHOICES OF PREDICTOR VARIABLES
X = data['baths']


# In[62]:


# DO NOT CHANGE, JUST RUN THE CELL
draw_scatter_plot(X, Y)


# In[13]:


# Predictor Variable - SET A SECOND PREDICTOR VARIABLE
X = data['sq__ft']


# In[14]:


# DO NOT CHANGE, JUST RUN THE CELL
draw_scatter_plot(X, Y)


# In[15]:


# Predictor Variable - SET A THIRD PREDICTOR VARIABLE
X = data['beds']


# In[16]:


# DO NOT CHANGE, JUST RUN THE CELL
draw_scatter_plot(X, Y)


# In[17]:


# Out of the three what do you think is the best predictor of price?

## sq_feet because it is scattered around many different values, 
## or in other words has a significant correlation with price


# #### Finding the best linear fit for the data to predict price.

# In[18]:


# Why does calculating mean won't work?

## Enter your reasoning, why mean or anyother singular measure of central tendency won't work here.
##The amount of data is so large and the amount of 0 sq ft entries will sqew the data
##


# ###### Let's analyse the Mean further.
# Considering the easiest prediction of price: Mean.
# 
# We have a number of houses, the easiest value to estimate the value of a new house in the area will be the mean.

# In[19]:


# Mean of price - FILL IN THE MISSING VALUE IN THE CODE TO CALCULATE THE MEAN OF PRICE
data['meanValues'] = data['price'].mean()


# In[20]:


# DO NOT CHANGE, JUST RUN THE CELL
X = data['sq__ft'] # lets consider the sq__ft values for a moment
Y = data['price']
meanValues = data['meanValues']


# In[21]:


# DO NOT CHANGE, JUST RUN THE CELL
def draw_plot(X, Y, Yhat):
    fig = plt.figure(figsize=(15,7))
    ax = plt.gca()
    ax.scatter(X, Y, c='b')
    ax.plot(X, Yhat, color='r');
    plt.show();


# In[22]:


# DO NOT CHANGE, JUST RUN THE CELL
draw_plot(X, Y, meanValues)


# In[23]:


# WHAT DO YOU THINK IS WRONG WITH THE GRAPH ABOVE?

## Type your analysis here.
#at 0 sqft there is still a high price on the property which makes no sense as they are paying ovr 800000 dollars for nothing 
# also the trend line is flat at a certain point showing no trend 


# #### Calculating the residuals.
# 
# Residual is basically the difference between the actual value and the predicted value. Therefore, the lesser the residual, the better the prediction.

# In[24]:


# ANSWER - Calculating the mean of the residual values - COMPLETE THE CODE TO CALCULATE THE RESIDUAL 
# BETWEEN THE PRICES AND MEAN
residual = abs(data['price'] - data['sq__ft']).mean()
residual


# But this is bad, why?
# 
# Because, only by looking at the data we can say that the price increases as the sq__ft values increase. But, the mean 
# value suggests that hte price remains the same with any value of the sq__ft. Plus, the average residual value is too high.
# 
# What is a better way?

# #### Predict Using Linear Regression
# 
# [Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line)
# 
# ### $$\hat{y} = \beta_0 + \beta_1 x$$
# 
# > ### $$ \beta_1 = \frac{\sum_{i=1}^n (y_i - \bar{y} ) (x_i - \bar{x} )}{\sum_{i=1}^n (x_i - \bar{x})^2} $$
# 
# and
# 
# > ### $$ \beta_0 = \bar{y} - \beta_1\bar{x} $$
# 
# Given variables:
# - $\bar{y}$ : the sample mean of observed values $Y$
# - $\bar{x}$ : the sample mean of observed values $X$
# - $s_Y$ : the sample standard deviation of observed values $Y$
# - $s_X$ : the sample standard deviation of observed values $X$
# - $r_{XY}$ : the sample Pearson correlation coefficient between observed $X$ and $Y$

# #### Defining X and Y.

# >>>>>X1

# In[25]:


# Predictor Variable - SET THE PREDICTOR VARIBLE (X1) TO ONE OF YOUR CHOICES OF PREDICTOR VARIABLES
X1 = data['sq__ft']
Y = data['price']


# #### Calculating beta1.

# In[26]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatebeta1(X, Y):
    Xbar = X.mean()
    Ybar = Y.mean()
    n = np.sum((Y - Ybar) * (X - Xbar)) 
    d = np.sum(np.square(X - Xbar)) 
    beta_1 = n/d
    return beta_1


# In[27]:


# Complete the code to calculate the beta1 for X1
beta1 = calculatebeta1(X1, Y)
beta1


# #### Calculating beta0.

# In[28]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatebeta0(X, Y):
    Xbar = X.mean()
    Ybar = Y.mean()
    beta0 = Ybar - calculatebeta1(X, Y) * Xbar
    return beta0


# In[29]:


# Complete the code to calculate the beta0 for X1
beta0 = calculatebeta0(X1, Y)
beta0


# #### Find the predicted values predictedValues.

# ### $$\hat{y} = \beta_0 + \beta_1 x$$

# In[30]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatePredictions(x, b0, b1):
    return b0 + b1 * x


# In[32]:


# Complete the code to calculate the preictions of values in X1
predictedValues1 = calculatePredictions(X1, 162930.16140699777,54.158639688530755)
predictedValues1


# #### Calculating the residuals.

# In[34]:


# Complete the code to calculate the residual for X1
residual = abs(data['sq__ft'] - (predictedValues1)).mean()
residual


# #### Plot of predicted values
# 
# The red line in the plot below shows the regression line calculated by the linear regression algorithm above.

# In[35]:


# DO NOT CHANGE, JUST RUN THE CELL
draw_plot(X1, Y, predictedValues1)


# >>>>>X2

# In[36]:


# Predictor Variable - SET THE PREDICTOR VARIBLE (X2) TO ANOTHER OF YOUR CHOICES OF PREDICTOR VARIABLES
X2 = data['beds']
Y = data['price']


# #### Calculating beta1.

# In[37]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatebeta1(X, Y):
    Xbar = X.mean()
    Ybar = Y.mean()
    n = np.sum((Y - Ybar) * (X - Xbar)) 
    d = np.sum(np.square(X - Xbar)) 
    beta_1 = n/d
    return beta_1


# In[38]:


# Complete the code to calculate the beta1 for X2
beta1 = calculatebeta1(X2, Y)
beta1


# #### Calculating beta0.

# In[39]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatebeta0(X, Y):
    Xbar = X.mean()
    Ybar = Y.mean()
    beta0 = Ybar - calculatebeta1(X, Y) * Xbar
    return beta0


# In[41]:


# Complete the code to calculate the beta0 for X2
beta0 = calculatebeta0(X2, Y)
beta0


# #### Find the predicted values predictedValues.

# ### $$\hat{y} = \beta_0 + \beta_1 x$$

# In[ ]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatePredictions(x, b0, b1):
    return b0 + b1 * x


# In[42]:


# Complete the code to calculate the predictions of values in X2
predictedValues2 = calculatePredictions(X2, 128792.15948942926, 36182.6439689373)
predictedValues2


# #### Calculating the residuals.

# In[43]:


# Complete the code to calculate the residual for X2
residual = abs(data['beds'] - (predictedValues2)).mean()
residual


# #### Plot of predicted values
# 
# The red line in the plot below shows the regression line calculated by the linear regression algorithm above.

# In[44]:


# DO NOT CHANGE, JUST RUN THE CELL
draw_plot(X2, Y, predictedValues2)


# >>>>>X3

# In[63]:


# Predictor Variable - SET THE PREDICTOR VARIBLE (X3) TO ANOTHER OF YOUR CHOICES OF PREDICTOR VARIABLES
X3 = data['baths']
Y = data['price']


# #### Calculating beta1.

# In[64]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatebeta1(X, Y):
    Xbar = X.mean()
    Ybar = Y.mean()
    n = np.sum((Y - Ybar) * (X - Xbar)) 
    d = np.sum(np.square(X - Xbar))
    beta_1 = n/d
    return beta_1


# In[65]:


# Complete the code to calculate the beta1 for X3
beta1 = calculatebeta1(X3, Y)
beta1


# #### Calculating beta0.

# In[66]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatebeta0(X, Y):
    
    Xbar = X.mean()
    Ybar = Y.mean()
    
    beta0 = Ybar - calculatebeta1(X, Y) * Xbar
    
    return beta0


# In[67]:


# Complete the code to calculate the beta0 for X3
beta0 = calculatebeta0(X3, Y)
beta0


# #### Find the predicted values predictedValues.

# ### $$\hat{y} = \beta_0 + \beta_1 x$$

# In[68]:


# DO NOT CHANGE, JUST RUN THE CELL
def calculatePredictions(x, b0, b1):
    return b0 + b1 * x


# In[69]:


# Complete the code to calculate the predictions of values in X3
predictedValues3 = calculatePredictions(X3, 119872.75465554741, 64318.535236734744)
predictedValues3


# #### Calculating the residuals.

# In[70]:


# Complete the code to calculate the residual for X3
residual = abs(data['baths'] - (predictedValues3)).mean()
residual


# #### Plot of predicted values
# 
# The red line in the plot below shows the regression line calculated by the linear regression algorithm above.

# In[71]:


# DO NOT CHANGE, JUST RUN THE CELL
draw_plot(X3, Y, predictedValues3)


# **Based on the three residual values which predictor provides the best predictions, X1, X2 or X3? Explain.**

# In[ ]:


#**Write your answers here.**
# X1 does because there is a better correlation and it shows the residual values way better


# ### 4) Predict price for a new house
# 
# For the given information:
# 
# - street:	1140 EDMONTON DR
# - city:	SACRAMENTO
# - zip:	95833
# - state:	CA
# - beds:	3
# - baths:	2
# - sq__ft:	1204
# - type:	Residential
# 
# **make a prediction for the house details given.**

# In[ ]:


x = 


# In[ ]:


ypredicted = 


# In[ ]:


print(ypredicted)


# In[ ]:




