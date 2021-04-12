#!/usr/bin/env python
# coding: utf-8

# ### A Look at the Data
# 
# In order to get a better understanding of the data we will be looking at throughout this lesson, let's take a look at some of the characteristics of the dataset.
# 
# First, let's read in the data and necessary libraries.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ALookAtTheData as t
from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./survey_results_public.csv')
df.head()


# As you work through the notebook(s) in this and future parts of this program, you will see some consistency in how to test your solutions to assure they match what we achieved!  In every environment, there is a solution file and a test file.  There will be checks for each solution built into each notebook, but if you get stuck, you may also open the solution notebook to see how we find any of the solutions.  Let's take a look at an example.
# 
# ### Question 1
# 
# **1.** Provide the number of rows and columns in this dataset.

# In[3]:


# We solved this one for you by providing the number of rows and columns:
# You can see how we are prompted that we solved for the number of rows and cols correctly!

num_rows = df.shape[0] #Provide the number of rows in the dataset
num_cols = df.shape[1] #Provide the number of columns in the dataset


# In[4]:


# If we made a mistake - a different prompt will appear

flipped_num_rows = df.shape[1] #Provide the number of rows in the dataset
flipped_num_cols = df.shape[0] #Provide the number of columns in the dataset


# Now that you are familiar with how to test your code - let's have you answer your first question:
# 
# ### Question 2
# 
# **2.** Which columns had no missing values? Provide a set of column names that have no missing values.

# In[6]:


no_nulls = set(df.columns[df.isnull().mean()==0])#Provide a set of columns with 0 missing values.


# ### Question 3
# 
# **3.** Which columns have the most missing values?  Provide a set of column names that have more than 75% if their values missing.

# In[7]:


most_missing_cols = set(df.columns[df.isnull().mean() > 0.75])#Provide a set of columns with more than 75% of the values missing


# ### Question 4
# 
# **4.** Provide a pandas series of the different **Professional** status values in the dataset along with the count of the number of individuals with each status.  Store this pandas series in **status_vals**.  If you are correct, you should see a bar chart of the proportion of individuals in each status.

# In[8]:


status_vals = df.Professional.value_counts()#Provide a pandas series of the counts for each Professional status

# The below should be a bar chart of the proportion of individuals in each professional category if your status_vals
# is set up correctly.
(status_vals/df.shape[0]).plot(kind='bar');
# (status_vals/df.shape[0]).plot(kind="bar");
plt.title("What kind of developer are you?");


# In[9]:


df.Professional.value_counts()


# ### Question 5
# 
# **5.** Provide a pandas series of the different **FormalEducation** status values in the dataset along with the count of how many individuals received that formal education.  Store this pandas series in **ed_vals**.  If you are correct, you should see a bar chart of the proportion of individuals in each status.

# In[10]:


ed_vals = df.FormalEducation.value_counts() #Provide a pandas series of the counts for each FormalEducation status

# The below should be a bar chart of the proportion of individuals in your ed_vals
# if it is set up correctly.

(ed_vals/df.shape[0]).plot(kind="bar");
plt.title("Formal Education");


# ### Question 6
# 
# **6.** Provide a pandas series of the different **Country** values in the dataset along with the count of how many individuals are from each country.  Store this pandas series in **count_vals**.  If you are correct, you should see a bar chart of the proportion of individuals in each country.

# In[11]:


count_vals = df.Country.value_counts() #Provide a pandas series of the counts for each Country

# The below should be a bar chart of the proportion of the top 10 countries for the
# individuals in your count_vals if it is set up correctly.

(count_vals[:10]/df.shape[0]).plot(kind="bar");
plt.title("Country");


# 

# In[ ]:





# In[ ]:




