#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


dataset=pd.read_csv('Position_Salaries.csv')


# In[5]:


dataset.head()


# In[6]:


X=dataset.iloc[:,1].values#create a matrix X and Y
Y=dataset.iloc[:,2].values
#we should convert X vector to matrix form
X=dataset.iloc[:,1:2].values#only on column so vector


# In[7]:


plt.scatter(X,Y)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[8]:


#first we will try simple linear regression
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(X,Y)#tranning model


# In[9]:


plt.scatter(X,Y)
plt.plot(X,linear.predict(X),color='red')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[10]:


from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2)
X_poly=poly_features.fit_transform(X)


# In[11]:


X_poly


# In[12]:


linear_poly=LinearRegression()
linear_poly.fit(X_poly,Y)


# In[13]:


plt.scatter(X,Y)
plt.plot(X,linear_poly.predict(poly_features.fit_transform(X)),color='red')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[14]:


poly_features=PolynomialFeatures(degree=4)
X_poly=poly_features.fit_transform(X)
linear_poly.fit(X_poly,Y)


# In[15]:


plt.scatter(X,Y)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.plot(X,linear_poly.predict(poly_features.fit_transform(X)),color='red')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[23]:


linear.predict(X_grid)


# In[20]:


linear_poly.predict(poly_features.fit_transform(X_grid))


# In[ ]:




