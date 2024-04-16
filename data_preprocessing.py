#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[2]:


df = pd.read_csv(r'F:\anu\infosys_internship\reviews.csv')
df.head()


# In[3]:


type(df)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


for i in range(5):
    print("Review: ", [i])
    print(df['content'].iloc[i], "\n")


# # converting into lower case

# In[7]:


df['content']=df['content'].str.lower()


# In[8]:


df.head()


# # removing links

# In[9]:


df['content'] = df['content'].fillna('')
df['content'] = df['content'].apply(lambda x: re.sub(r"http\S+", "", x))


# In[10]:


df.head()


# # Remove next lines
# 

# In[11]:


df['content'] = df['content'].str.replace('\n',' ')


# In[12]:


df.head()


# # removing words containing numbers

# In[29]:


pattern = r'\b\w*\d\w*\b'
def remove_words_containing_numbers(text):
    return re.sub(pattern, '', text)

df['content'] = df['content'].apply(remove_words_with_numbers)
df.head()


# # removing extra spaces

# In[14]:


df['content'] = df['content'].apply(lambda x: ' '.join(x.split()))
df.head()


# # removing special characters 

# In[15]:


df['content'] = df['content'].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x))
df.head()


# # removing stop words

# In[16]:


nltk.download('stopwords')


# In[17]:


df.info()


# In[18]:


stop_words = set(stopwords.words('english'))
df['content']= df['content'].apply(lambda x: ' '.join(word for word in x.split() if word.lower() not in stop_words))


# In[19]:


df.head()


# # stemming process

# In[20]:


get_ipython().system('pip install -U nltk')


# In[21]:


stemmer = PorterStemmer()
df['content'] = df['content'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


# In[22]:


df.head()


# In[23]:


for i in range(5):
    print("Review: ", [i])
    print(df['content'].iloc[i], "\n")


# # lemmatization

# In[24]:


nltk.download('wordnet')


# In[25]:


nltk.download('omw-1.4')


# In[26]:


lemmatizer = WordNetLemmatizer()
df['content'] = df['content'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))


# In[27]:


df.head()


# In[28]:


for i in range(5):
    print("Review: ", [i])
    print(df['content'].iloc[i], "\n")

