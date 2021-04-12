#!/usr/bin/env python
# coding: utf-8

# #### Removing Values
# 
# You have seen:
# 
# 1. sklearn break when introducing missing values
# 2. reasons for dropping missing values
# 
# It is time to make sure you are comfortable with the methods for dropping missing values in pandas.  You can drop values by row or by column, and you can drop based on whether **any** value is missing in a particular row or column or **all** are values in a row or column are missing.

# In[2]:


import numpy as np
import pandas as pd
import RemovingValues as t
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

small_dataset = pd.DataFrame({'col1': [1, 2, np.nan, np.nan, 5, 6], 
                              'col2': [7, 8, np.nan, 10, 11, 12],
                              'col3': [np.nan, 14, np.nan, 16, 17, 18]})

small_dataset


# #### Question 1
# 
# **1.** Drop any row with a missing value.

# In[3]:


all_drop  = small_dataset.dropna()# Drop any row with a missing value


#print result
all_drop


# #### Question 2
# 
# **2.** Drop only the row with all missing values.

# In[6]:


all_row = small_dataset.dropna(axis=0, how="all")# Drop only rows with all missing values 


#print result
all_row


# #### Question 3
# 
# **3.** Drop only the rows with missing values in column 3.

# In[7]:


only3_drop = small_dataset.dropna(subset= ["col3"], how="any")# Drop only rows with missing values in column 3

#print result
only3_drop


# #### Question 4
# 
# **4.** Drop only the rows with missing values in column 3 or column 1.

# In[8]:


only3or1_drop = small_dataset.dropna(subset=["col1", "col3"], how="any")# Drop rows with missing values in column 1 or column 3


#print result
only3or1_drop


# In[ ]:




