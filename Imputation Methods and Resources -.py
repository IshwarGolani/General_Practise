#!/usr/bin/env python
# coding: utf-8

# 
# ### Imputation Methods and Resources
# 
# One of the most common methods for working with missing values is by imputing the missing values.  Imputation means that you input a value for values that were originally missing. 
# 
# It is very common to impute in the following ways:
# 1. Impute the **mean** of a column.<br><br>
# 
# 2. If you are working with categorical data or a variable with outliers, then use the **mode** of the column.<br><br>
# 
# 3. Impute 0, a very small number, or a very large number to differentiate missing values from other values.<br><br>
# 
# 4. Use knn to impute values based on features that are most similar.<br><br>
# 
# In general, you should try to be more careful with missing data in understanding the real world implications and reasons for why the missing values exist.  At the same time, these solutions are very quick, and they enable you to get models off the ground.  You can then iterate on your feature engineering to be more careful as time permits.
# 
# Create the dataset you will be using for this notebook using the code below.
# 

# In[1]:


import pandas as pd
import numpy as np
import ImputationMethods as t

df = pd.DataFrame({'A':[np.nan, 2, np.nan, 0, 7, 10, 15],
                   'B':[3, 4, 5, 1, 2, 3, 5],
                   'C':[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                   'D':[np.nan, True, np.nan, False, True, False, np.nan],
                   'E':['Yes', 'No', 'Maybe', np.nan, np.nan, 'Yes', np.nan]})

df.head()


# #### Question 1
# 
# **1.** Use the dictionary below to label the columns as the appropriate data type.

# In[2]:


a = 'categorical'
b = 'quantitative'
c = 'we cannot tell'
d = 'boolean - can treat either way'

question1_solution = {'Column A is':b, #letter here,
                      'Column B is':b, #letter here,
                      'Column C is':c,#letter here,
                      'Column D is':d, #letter here,
                      'Column E is':a #letter here
                     }


# #### Question 2
# 
# **2.** Are there any columns or rows that you feel comfortable dropping in this dataframe?

# In[3]:


a = "Yes"
b = "No"

should_we_drop = a


# In[4]:


new_df = df.drop('C',axis=1)
# Use this cell to drop any columns or rows you feel comfortable dropping based on the above
# new_df.head()


# #### Question 3
# 
# **3.** Using **new_df**, I wrote a lambda function that you can use to impute the mean for the columns of your dataframe using the **apply** method.  Use as many cells as you need to correctly fill in the dictionary **impute_q3** to answer a few questions about your findings.

# In[5]:


fill_mean = lambda col: col.fillna(col.mean())
# fill_mean = lambda col: col.fillna(col.mean()[0])
try:
#     new_df.apply(fill_mean, axis=0)
    new_df.apply(fill_mean, axis =0)
except:
    print('That broke...because cloumn E has string')


# In[6]:


new_df[['A','B','D']].apply(fill_mean, axis = 0)# Check what you need to answer the questions below


# In[8]:


a = "fills with the mean, but that doesn't actually make sense in this case."
b = "gives an error."
c = "is no problem - it fills the NaN values with the mean as expected."


impute_q3 = {'Filling column A': c, #letter here,
             'Filling column D':a, #letter here,
             'Filling column E':b #letter here    
}


# #### Question 4
# 
# **4.** Given the results above, it might make more sense to fill some columns with the mode.  Write your own function to fill a column with the mode value, and use it on the two columns that might benefit from this type of imputation.  Use the dictionary **impute_q4** to answer some questions about your findings.

# In[9]:


fill_mode = lambda col: col.fillna(col.mode()[0])#Similar to the above write a function and apply it to compte the mode for each column
new_df.apply(fill_mode, axis=0)
#If you get stuck, here is a helpful resource https://stackoverflow.com/questions/42789324/pandas-fillna-mode


# In[10]:


# new_df.head()


# In[12]:


a = "Did not impute the mode."
b = "Imputes the mode."


impute_q4 = {'Filling column A': a, #letter here,
             'Filling column D':a, #letter here,
             'Filling column E':b #letter here
            }


# You saw two of the most common ways to impute values in this notebook, and hopefully, you realized that even these methods have complications.  Again, these methods can be a great first step to get your models off the ground, but there are potentially detrimental aspects to the bias introduced into your models using these methods.

# In[ ]:




