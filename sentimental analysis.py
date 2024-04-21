#!/usr/bin/env python
# coding: utf-8

# In[3]:


#libraries
import pandas as pd
import re
from bs4 import BeautifulSoup
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[8]:


#reading the csv
df = pd.read_csv(r'C:\Users\AAKASH\OneDrive\Documents\internship\reviews.csv')


# In[9]:


#printing actual column data(reviews)
df['content'].head()


# In[10]:


#converting text into lower case
df['lw_content'] = df['content'].apply(lambda x: x.lower())


# In[11]:


#removing non textual content
def remove_non_textual(text):
    return re.sub(r'[^\w\s]', '', text)
df['nt_content'] = df['lw_content'].apply(remove_non_textual)


# In[12]:


# to strip HTML tags 
def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()
df['nonhtml_text'] = df['nt_content'].apply(remove_html_tags)


# In[13]:


# to expand contractions 
df['ct_data'] = df['nonhtml_text'].apply(lambda x: contractions.fix(x))


# In[14]:


#removing stop words
stopwords.words('english')
[punc for punc in string.punctuation]
def text_process(msg):
  nopunc= [char for char in msg if char not in string.punctuation]
  nopunc= ''.join(nopunc)
  return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])
df['cleaned_data']= df['ct_data'].apply(text_process)


# In[15]:


#printing cleaned data
df['cleaned_data'].head()


# In[16]:


#lemmatizing of data
lemmatizer = WordNetLemmatizer()
#Function to lemmatize a sentence
def lemmatize_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)  # Tokenize the sentence into words
    lemmatized_sentence = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens])
    return lemmatized_sentence
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()  # Get the POS tag
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) 


# In[17]:


df['lemmatized_text'] = df['cleaned_data'].apply(lemmatize_sentence)


# In[18]:


df['lemmatized_text'].head()


# In[19]:


#Word Cloud
textvv = ' '.join(df['lemmatized_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(textvv)

# Ploting Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[21]:


#sentiment analysis
sid = SentimentIntensityAnalyzer()
def vader_sentiment_analysis(text):
    scores = sid.polarity_scores(text)
    return scores['compound']


# In[29]:


df['sanalysis_score'] = df['lemmatized_text'].apply(vader_sentiment_analysis)


# In[30]:


print(df[['content', 'sanalysis_score']])


# In[ ]:




