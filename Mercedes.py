#!/usr/bin/env python
# coding: utf-8

# # Mercedes-Benz Greener Manufacturing

# DESCRIPTION
# 
# Reduce the time a Mercedes-Benz spends on the test bench.
# 
# Problem Statement Scenario:
# Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.
# 
# To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.
# 
# You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Mercedes_train.csv')
df_test=pd.read_csv('Mercedes_test.csv')
df


# ## If for any column(s), the variance is equal to zero, then you need to remove those variable(s).

# In[3]:


from statistics import variance


# In[4]:


dfn=df.select_dtypes(include=np.number)
dfn


# In[5]:


variance(dfn[dfn.columns[1]])


# In[6]:


cols=[]
for col in dfn.columns:
    if variance(dfn[col]) == 0:
        print(col,': ',variance(dfn[col]))
        cols +=[col]


# In[7]:


df=df.drop(columns=cols)


# In[8]:


df_test=df_test.drop(columns=cols)


# ## Check for null and unique values for test and train sets.

# In[9]:


na_counts=df.isna().sum()
na_counts[na_counts>0]


# ## Apply label encoder.

# In[10]:


cat=list(df.dtypes[df.dtypes=='object'].index)
cat


# In[11]:


df['X0'].unique()


# In[12]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()


# In[13]:


for c in cat:
    df[c]= label_encoder.fit_transform(df[c])
    df_test[c]= label_encoder.fit_transform(df_test[c])


# In[14]:


df[cat]


# ## Perform dimensionality reduction.

# In[15]:


df.apply(lambda x: x.nunique()).sort_values()


# In[16]:


df


# In[17]:


set(df.columns)- set(['ID', 'y'])


# In[18]:


usable_columns = list(set(df.columns) - set(['ID', 'y']))
y_train = df['y'].values
id_test = df_test['ID'].values

x_train = df[usable_columns]
x_test = df_test[usable_columns]


# ## Predict your test_df values using XGBoost.

# In[25]:


n_comp = 12
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(x_train)
pca2_results_test = pca.transform(x_test)


# In[26]:


import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(
        pca2_results_train, 
        y_train, test_size=0.2, 
        random_state=4242)


# In[27]:


d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
#d_test = xgb.DMatrix(x_test)
d_test = xgb.DMatrix(pca2_results_test)


# In[28]:


params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4


# In[29]:


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


# In[30]:


watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 
                1000, watchlist, early_stopping_rounds=50, 
                feval=xgb_r2_score, maximize=True, verbose_eval=10)


# In[31]:


# Step12: Predict your test_df values using xgboost

p_test = clf.predict(d_test)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = p_test


# In[32]:


sub

