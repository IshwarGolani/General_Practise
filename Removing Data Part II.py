#!/usr/bin/env python
# coding: utf-8

# #### Removing Data Part II
# 
# So, you now have seen how we can fit a model by dropping rows with missing values.  This is great in that sklearn doesn't break! However, this means future observations will not obtain a prediction if they have missing values in any of the columns.
# 
# In this notebook, you will answer a few questions about what happened in the last screencast, and take a few additional steps.

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import RemovingData as t
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./survey_results_public.csv')

#Subset to only quantitative vars
num_vars = df[['Salary', 'CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]


num_vars.head()


# #### Question 1
# 
# **1.** What proportion of individuals in the dataset reported a salary?

# In[6]:


prop_sals = 1- df.isnull()["Salary"].mean() # Proportion of individuals in the dataset with salary reported

prop_sals


# #### Question 2
# 
# **2.** Remove the rows associated with nan values in Salary (only Salary) from the dataframe **num_vars**.  Store the dataframe with these rows removed in **sal_rem**.

# In[7]:


sal_rm = num_vars.dropna(subset= ["Salary"], axis= 0) # dataframe with rows for nan Salaries removed

sal_rm.shape


# #### Question 3
# 
# **3.** Using **sal_rm**, create **X** be a dataframe (matrix) of all of the numeric feature variables.  Then, let **y** be the response vector you would like to predict (Salary).  Run the cell below once you have split the data, and use the result of the code to assign the correct letter to **question3_solution**.

# In[8]:


X = sal_rm[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]#Create X using explanatory variables from sal_rm
y = sal_rm['Salary']#Create y using the response variable of Salary

# Split data into training and test data, and fit a linear model
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=.30, random_state=42)
lm_model = LinearRegression(normalize=True)

# If our model works, it should just fit our model to the data. Otherwise, it will let us know.
try:
    lm_model.fit(X_train, y_train)
except:
    print("Oh no! It doesn't work!!!")


# In[9]:


a = 'Python just likes to break sometimes for no reason at all.' 
b = 'It worked, because Python is magic.'
c = 'It broke because we still have missing values in X'

question3_solution = c #Letter here


# #### Question 4
# 
# **4.** Remove the rows associated with nan values in any column from **num_vars** (this was the removal process used in the screencast).  Store the dataframe with these rows removed in **all_rem**.

# In[10]:


all_rm = num_vars.dropna(axis=0)# dataframe with rows for nan Salaries removed

all_rm.head()


# #### Question 5
# 
# **5.** Using **all_rm**, create **X_2** be a dataframe (matrix) of all of the numeric feature variables.  Then, let **y_2** be the response vector you would like to predict (Salary).  Run the cell below once you have split the data, and use the result of the code to assign the correct letter to **question5_solution**.

# In[11]:


X_2 = all_rm[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]#Create X using explanatory variables from all_rm
y_2 = all_rm["Salary"] #Create y using Salary from sal_rm

# Split data into training and test data, and fit a linear model
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2 , test_size=.30, random_state=42)
lm_2_model = LinearRegression(normalize=True)

# If our model works, it should just fit our model to the data. Otherwise, it will let us know.
try:
    lm_2_model.fit(X_2_train, y_2_train)
except:
    print("Oh no! It doesn't work!!!")


# In[12]:


a = 'Python just likes to break sometimes for no reason at all.' 
b = 'It worked, because Python is magic.'
c = 'It broke because we still have missing values in X'

question5_solution = b#Letter here


# #### Question 6
# 
# **6.** Now, use **lm_2_model** to predict the **y_2_test** response values, and obtain an r-squared value for how well the predicted values compare to the actual test values.  

# In[13]:


y_test_preds = lm_2_model.predict(X_2_test)# Predictions here using X_2 and lm_2_model
r2_test = r2_score(y_2_test, y_test_preds) # Rsquared here for comparing test and preds from lm_2_model

# Print r2 to see result
r2_test


# In[14]:


y_test_preds.shape


# #### Question 7
# 
# **7.** Use what you have learned **from the second model you fit** (and as many cells as you need to find the answers) to complete the dictionary with the variables that link to the corresponding descriptions.

# In[15]:


a = 5009
b = 'Other'
c = 645
d = 'We still want to predict their salary'
e = 'We do not care to predict their salary'
f = False
g = True

question7_solution = {'The number of reported salaries in the original dataset': a, #Letter here,
                       'The number of test salaries predicted using our model':c, #Letter here,
                       'If an individual does not rate stackoverflow, but has a salary':d, #Letter here,
                       'If an individual does not have a a job satisfaction, but has a salary':d, #Letter here,
                       'Our model predicts salaries for the two individuals described above.':f} #Letter here}
                      
                      


# In[16]:


print("The number of salaries in the original dataframe is " + str(np.sum(df.Salary.notnull()))) 
print("The number of salaries predicted using our model is " + str(len(y_test_preds)))
print("This is bad because we only predicted " + str((len(y_test_preds))/np.sum(df.Salary.notnull())) + " of the salaries in the dataset.")


# In[17]:


print("The number of salaries in the original dataframe is " + str(np.sum(df.Salary.notnull())))
# print("The number of salaries predicted using our model is " + str(len(y_test_preds)))
# print("This is bad because we only predicted " + str((len(y_test_preds))/np.sum(df.Salary.notnull())) + " of the salaries in the dataset.")rk


# In[18]:


print("The number of salaries predicted using our model is " + str(len(y_test_preds)))


# In[19]:


print("This is bad because we only predicted " + str(r2_test) + " of the salaries in the dataset")


# In[ ]:




