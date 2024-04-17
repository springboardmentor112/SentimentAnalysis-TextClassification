#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set seed for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv("reviews.csv") 


# In[32]:


# Load the dataset
data = pd.read_csv("reviews.csv")

print(data.head())


# In[33]:


# Converted text to lowercase
data['content'] = data['content'].str.lower()
data['replyContent'] = data['replyContent'].str.lower()

# Printed the first few rows to verify the changes
print(data.head())


# In[35]:


# Removed newline characters
data['content'] = data['content'].replace('\n', ' ')
data['replyContent'] = data['replyContent'].replace('\n', ' ')

# Printed the first few rows to verify the changes
print(data.head())


# In[36]:


# Removed extra spaces
data['content'] = data['content'].str.replace(' +', ' ')
data['replyContent'] = data['replyContent'].str.replace(' +', ' ')

# Printed the first few rows to verify the changes
print(data.head())


# In[38]:


#Removed special characters
data['content'] = data['content'].str.replace(r'[^\w\s]', '')
data['replyContent'] = data['replyContent'].str.replace(r'[^\w\s]', '')

# Printed the first few rows to verify the changes
print(data.head())


# In[40]:


import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("reviews.csv")

# Convert NaN values to empty strings
data['content'] = data['content'].fillna('')
data['replyContent'] = data['replyContent'].fillna('')

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in str(text).split() if word.lower() not in stop_words])

# Apply the function to the 'content' and 'replyContent' columns
data['content'] = data['content'].apply(remove_stopwords)
data['replyContent'] = data['replyContent'].apply(remove_stopwords)

# Print the first few rows to verify the changes
print(data.head())


# In[41]:


from nltk.stem import WordNetLemmatizer

# Download WordNet Lemmatizer (run once)
nltk.download('wordnet')

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Function to apply lemmatization
def apply_lemmatization(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Apply lemmatization to the 'content' and 'replyContent' columns
data['content'] = data['content'].apply(apply_lemmatization)
data['replyContent'] = data['replyContent'].apply(apply_lemmatization)

# Print the first few rows to verify the changes
print(data.head())


# In[42]:


from nltk.stem import PorterStemmer

# Create a stemmer object
stemmer = PorterStemmer()

# Function to apply stemming
def apply_stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Apply stemming to the 'content' and 'replyContent' columns
data['content'] = data['content'].apply(apply_stemming)
data['replyContent'] = data['replyContent'].apply(apply_stemming)

# Print the first few rows to verify the changes
print(data.head())

