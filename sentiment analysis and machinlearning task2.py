#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#The lexicon approach in sentiment analysis involves using predefined lexicons or dictionaries of words and their
#associated sentiment scores to determine the sentiment expressed in text data. 


# In[ ]:


#this apprach can only done after cleaning the dataset then we will have new clean  dataset in which we need to perform 
#sentiment analysis


# In[ ]:


import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the CSV dataset
df = pd.read_csv('reviews.csv')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to classify sentiment
def classify_sentiment(text):
    text = str(text)
    # Perform sentiment analysis using VADER
    sentiment_scores = sid.polarity_scores(text)
    # Determine sentiment category based on compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment classification to a specific column of the dataset
df['Lexical'] = df['Content'].apply(classify_sentiment)


# In[ ]:


df.head()


# In[ ]:


#machine learning for sentiment analysis second approach


# In[ ]:


#We'll use a popular supervised learning algorithm, 
#such as Logistic Regression, along with TF-IDF vectorization for feature extraction.


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('reviews.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features parameter
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model Building: Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_tfidf, y_train)

# Predictions
y_pred_train = logistic_regression.predict(X_train_tfidf)
y_pred_test = logistic_regression.predict(X_test_tfidf)

# Evaluation
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

print("\nClassification Report for Testing Set:")
print(classification_report(y_test, y_pred_test))


# In[ ]:




