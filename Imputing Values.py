#!/usr/bin/env python
# coding: utf-8

# #### Imputing Values
# 
# You now have some experience working with missing values, and imputing based on common methods.  Now, it is your turn to put your skills to work in being able to predict for rows even when they have NaN values.
# 
# First, let's read in the necessary libraries, and get the results together from what you achieved in the previous attempt.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import ImputingValues as t
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./survey_results_public.csv')
df.head()

#Only use quant variables and drop any rows with missing values
num_vars = df[['Salary', 'CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
df_dropna = num_vars.dropna(axis=0)

#Split into explanatory and response variables
X = df_dropna[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
y = df_dropna['Salary']

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 

lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit
        
#Predict and score the model
y_test_preds = lm_model.predict(X_test) 
"The r-squared score for your model was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))


# #### Question 1
# 
# **1.** As you may remember from an earlier analysis, there are many more salaries to predict than the values shown from the above code.  One of the ways we can start to make predictions on these values is by imputing items into the **X** matrix instead of dropping them.
# 
# Using the **num_vars** dataframe drop the rows with missing values of the response (Salary) - store this new dataframe in **drop_sal_df**, then impute the values for all the other missing values with the mean of the column - store this in **fill_df**.

# In[2]:


drop_sal_df = num_vars.dropna(subset=['Salary'], axis=0)#Drop the rows with missing salaries

# test look
drop_sal_df.head()


# In[3]:


fill_mean = lambda col: col.fillna(col.mean())#Fill all missing values with the mean of the column.

fill_df = drop_sal_df.apply(fill_mean, axis=0)
# test look
fill_df.head()


# #### Question 2
# 
# **2.** Using **fill_df**, predict Salary based on all of the other quantitative variables in the dataset.  You can use the template above to assist in fitting your model:
# 
# * Split the data into explanatory and response variables
# * Split the data into train and test (using seed of 42 and test_size of .30 as above)
# * Instantiate your linear model using normalized data
# * Fit your model on the training data
# * Predict using the test data
# * Compute a score for your model fit on all the data, and show how many rows you predicted for
# 
# Use the tests to assure you completed the steps correctly.

# In[5]:


#Split into explanatory and response variables
X = fill_df[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
y = fill_df['Salary']
#Split into train and test
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
#Predict and score the model
lm_model = LinearRegression(normalize = True)
lm_model.fit(X_train,y_train)

y_test_preds = lm_model.predict(X_test)
#Rsquared and y_test
rsquared_score = r2_score(y_test, y_test_preds)#r2_score
length_y_test = len(y_test) #num in y_test

"The r-squared score for your model was {} on {} values.".format(rsquared_score, length_y_test)


# This model still isn't great.  Let's see if we can't improve it by using some of the other columns in the dataset.

# In[ ]:




