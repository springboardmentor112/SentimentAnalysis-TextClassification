#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
import spacy


# In[48]:


df= pd.read_csv('reviews.csv')


# In[49]:


df.head()


# # lower case

# In[27]:


df["content"] = df["content"].str.lower()


# # removinf links

# In[31]:


df['content'] = df['content'].fillna('')


# In[ ]:


no_url=[]
for sentence in data['content']:
    no_url.append(re.sub(r"http\S+", "", sentence))
df['content']=no_url


# # remove next line

# In[36]:


df['content'] = data['content'].str.replace('\n','')


# # removing extra space

# In[35]:


df['content'] = data['content'].apply(lambda x: ''.join(x.split()))


# # removing worlds containing number

# In[51]:


pattern = r'\b\w*\d\w*\b'
def remove_words_with_numbers(text):
    return re.sub(pattern, '',text)
df['content'] = df['content'].apply(remove_words_with_numbers)


# # removing special character

# In[38]:


no_special_char=[]
for sentence in data.content:
    no_special_char.append(re.sub('[A-Za-z0-9]+', ' ', sentence))
df['content']=no_special_char


# # removel of stopwords

# In[46]:


stop_words = set(stopwords.words('english'))


# In[40]:


def remove_stopwords(text):
    tokens = text.split()
    filtered_text = [word for word in tokens if word.lower() not in stop_words]
    return' '.join(filtered_text)


# In[48]:


df['content'] = data['content'].apply(remove_stopwords)


# # stemming

# In[66]:


stemmer = PorterStemmer()


# In[76]:


df['content'] =  df['content'].apply(lambda text:' '.join([stemmer.stem(word) for word in word_tokenize(text)]))


# # Lemmatization

# In[ ]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[ ]:


nlp = spacy.load('en_core_web_sm')


# In[ ]:


def lemmatize_text(text): doc = nlp(text) lemmatized_tokens = [token.lemma_ for token in doc] lemmatized_text = ' '.join(lemmatized_tokens) return lemmatized_text


# In[ ]:


df['content'] = df['content'].apply(lemmatizer_text)


# In[78]:


print(df['content'])


# In[9]:


df.to_csv('newreviwe')


# In[ ]:




