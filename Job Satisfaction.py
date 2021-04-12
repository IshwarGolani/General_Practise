#!/usr/bin/env python
# coding: utf-8

# #### Job Satisfaction
# 
# In this notebook, you will be exploring job satisfaction according to the survey results.  Use the cells at the top of the notebook to explore as necessary, and use your findings to solve the questions at the bottom of the notebook.

# In[2]:


import pandas as pd
import numpy as np
import JobSatisfaction as t
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./survey_results_public.csv')
schema = pd.read_csv('./survey_results_schema.csv')
df.head()


# In[3]:


df["JobSatisfaction"].isnull().mean()#Space for dfyour code


# In[4]:


df.groupby(["EmploymentStatus"]).mean()["JobSatisfaction"]#More space for code


# In[5]:


df.groupby(["CompanySize"]).mean()["JobSatisfaction"].sort_values()#Additional space for your additional code


# In[6]:


#Feel free to create new cells as you need them


# #### Question 1
# 
# **1.** Use the space above to assist in matching each variable (**a**, **b**, **c**, **d**, **e**, **f**, **g**, or **h** ) as the appropriate key that describes the value in the **job_sol_1** dictionary.

# In[7]:


a = 0.734
b = 0.2014
c = 'full-time'
d = 'contractors'
e = 'retired'
f = 'yes'
g = 'no'
h = 'hard to tell'

job_sol_1 = {'The proportion of missing values in the Job Satisfaction column':b, #letter here,
             'According to EmploymentStatus, which group has the highest average job satisfaction?':d, #letter here, 
             'In general, do smaller companies appear to have employees with higher job satisfaction?':f #letter here
            }


# #### Question 2
# 
# **2.** Use the space above to assist in matching each variable (**a**, **b**, **c** ) as the appropriate key that describes the value in the **job_sol_2** dictionary. Notice you can have the same letter appear more than once.

# In[8]:


df.groupby(["ProgramHobby"]).mean()["JobSatisfaction"].sort_values().dropna()


# In[9]:


df.groupby(["HomeRemote"]).mean()["JobSatisfaction"].sort_values().dropna()


# In[10]:


df.groupby(["FormalEducation"]).mean()["JobSatisfaction"].sort_values().dropna()


# In[11]:


a = 'yes'
b = 'no'
c = 'hard to tell'

job_sol_2 = {'Do individuals who program outside of work appear to have higher JobSatisfaction?': a,#letter here,
             'Does flexibility to work outside of the office appear to have an influence on JobSatisfaction?':a, #letter here, 
             'A friend says a Doctoral degree increases the chance of having job you like, does this seem true?':a} #letter here}
             


# In[ ]:




