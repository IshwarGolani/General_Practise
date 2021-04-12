#!/usr/bin/env python
# coding: utf-8

# ### First Try of Predicting Salary
# 
# For the last two questions regarding what are related to relationships of variables with salary and job satisfaction - Each of these questions will involve not only building some sort of predictive model, but also finding and interpretting the influential components of whatever model we build.
# 
# To get started let's read in the necessary libraries and take a look at some of our columns of interest.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import WhatHappened as t
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./survey_results_public.csv')
df.head()


# Now take a look at the summary statistics associated with the quantitative variables in your dataset. 

# In[2]:


df.describe()


# #### Question 1
# 
# **1.** Use the above to match each variable (**a**, **b**, **c**, **d**, **e**, or **f**) as the appropriate key that describes the value in the **desc_sol** dictionary.

# In[ ]:





# In[5]:


a = 40
b = 'HoursPerWeek'
c = 'Salary'
d = 'Respondent'
e = 10
f = 'ExpectedSalary'

desc_sol = {'A column just listing an index for each row':d,
       'The maximum Satisfaction on the scales for the survey':e,
       'The column with the most missing values':f, #letter here,
       'The variable with the highest spread of values':c}#letter here}

# Check your solution


# A picture can often tell us more than numbers.

# In[6]:


df.hist();


# Often a useful plot is a correlation matrix - this can tell you which variables are related to one another.

# In[7]:


# sns.heatmap(df.corr(), annot=True, fmt=".2f");
sns.heatmap(df.corr(), annot=True, fmt=".2f");


# #### Question 2
# 
# **2.** Use the scatterplot matrix above to match each variable (**a**, **b**, **c**, **d**, **e**, **f**, or **g**) as the appropriate key that describes the value in the **scatter_sol** dictionary.

# In[8]:


a = 0.65
b = -0.01
c = 'ExpectedSalary'
d = 'No'
e = 'Yes'
f = 'CareerSatisfaction'
g = -0.15

scatter_sol = {'The column with the strongest correlation with Salary':f, #letter here,
       'The data suggests more hours worked relates to higher salary': d, #letter here,
       'Data in the ______ column meant missing data in three other columns': c, #letter here,
       'The strongest negative relationship had what correlation?': g } #letter here}


# Here we move our quantitative variables to an X matrix, which we will use to predict our response.  We also create our response.  We then split our data into training and testing data.  Then when starting our four step process, our fit step breaks.  
# 
# ### Remember, this code will break!

# In[9]:


# Consider only numerica variables
X = df[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
y = df['Salary']

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = .30, random_state = 42 )
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

# #Four steps:

# #Instantiate
# lm_model = LinearRegression(normalize=True)
lm_model = LinearRegression(normalize=True) 
# lm_model.fit(X_train, y_train)
# #Fit - why does this break?
lm_model.fit(X_train, y_train) 

# #Predict
# #Score


# #### Question 3
# 
# **3.** Use the results above to match each variable (**a**, **b**, **c**, **d**, **e**, or **f** ) as the appropriate key that describes the value in the **lm_fit_sol** dictionary.

# In[11]:


a = 'it is a way to assure your model extends well to new data'
b = 'it assures the same train and test split will occur for different users'
c = 'there is no correct match of this question'
d = 'sklearn fit methods cannot accept NAN values'
e = 'it is just a convention people do that will likely go away soon'
f = 'python just breaks for no reason sometimes'

lm_fit_sol = {'What is the reason that the fit method broke?': d,#letter here,
       'What does the random_state parameter do for the train_test_split function?': b,#letter here,
       'What is the purpose of creating a train test split?':a} #letter here}


# In[ ]:




