#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[32]:

# https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
df = pd.read_csv('BankNote_Authentication.csv')
df.head()


# In[33]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[35]:


X.head()


# In[36]:


y.head()


# In[37]:


# Train Test Split
# Test size is 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


# In[38]:


# Implement Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# In[40]:


# Predict y based on thes X_test file
y_pred=classifier.predict(X_test)


# In[41]:


# check the accuracy of the prediction comparing to the actual y_test
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test, y_pred)
score


# In[42]:


# Creating a pickle file using serialization
# for API
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[43]:


# Let's see how the model predicts y based on the 4 values of X
import numpy as np
classifier.predict([[2,3,4,1]])

