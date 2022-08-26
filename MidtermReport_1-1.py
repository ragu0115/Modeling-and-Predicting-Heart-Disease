#!/usr/bin/env python
# coding: utf-8

# In[186]:


import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[187]:


# Importing Data
df = pd.read_csv('C:\\Users\\raghu\\Desktop\\heart.csv')
df


# In[207]:


df.info()


# In[188]:


# Dropping Duplicate Data
df.drop_duplicates(inplace=True)
#Splitting X and y Variables
X = df.drop(columns='HeartDiseaseorAttack')
y = pd.DataFrame(df['HeartDiseaseorAttack'])
# correlation heat map
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True,annot_kws={"size": 5},cmap="PuOr")


# In[215]:


df.plot(kind='box')


# In[196]:


# Linear Regression
linmodel = sm.OLS(y,sm.add_constant(X)).fit()
linmodel.summary()


# In[198]:


# Logistic Regression
from statsmodels.tools import add_constant as add_constant
df = add_constant(df)
st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols = df.columns[:-1]
model = sm.Logit(df.HeartDiseaseorAttack, df[cols])
result = model.fit()
result.summary()


# In[189]:


# Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify= y, random_state=31)
# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[202]:


# Testing Acccuracy of Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
classification_model = LogisticRegression(max_iter = 10000)
logmodel = classification_model.fit(X_train, y_train.values.ravel())
predictions = classification_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
accuracy

