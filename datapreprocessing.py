#!/usr/bin/env python
# coding: utf-8

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
reviews_df = pd.read_csv("reviews.csv") 

# Print the first few rows of the dataset
print(reviews_df.head())

# Convert text to lowercase
reviews_df['content'] = reviews_df['content'].str.lower()
reviews_df['replyContent'] = reviews_df['replyContent'].str.lower()

# Print the first few rows to verify the changes
print(reviews_df.head())

# Remove newline characters
reviews_df['content'] = reviews_df['content'].replace('\n', ' ')
reviews_df['replyContent'] = reviews_df['replyContent'].replace('\n', ' ')

# Print the first few rows to verify the changes
print(reviews_df.head())

# Remove extra spaces
reviews_df['content'] = reviews_df['content'].str.replace(' +', ' ')
reviews_df['replyContent'] = reviews_df['replyContent'].str.replace(' +', ' ')

# Print the first few rows to verify the changes
print(reviews_df.head())

# Remove special characters
reviews_df['content'] = reviews_df['content'].str.replace(r'[^\w\s]', '')
reviews_df['replyContent'] = reviews_df['replyContent'].str.replace(r'[^\w\s]', '')

# Print the first few rows to verify the changes
print(reviews_df.head())

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in str(text).split() if word.lower() not in stop_words])

# Apply the function to the 'content' and 'replyContent' columns
reviews_df['content'] = reviews_df['content'].apply(remove_stopwords)
reviews_df['replyContent'] = reviews_df['replyContent'].apply(remove_stopwords)

# Print the first few rows to verify the changes
print(reviews_df.head())

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Function to apply lemmatization
def apply_lemmatization(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Apply lemmatization to the 'content' and 'replyContent' columns
reviews_df['content'] = reviews_df['content'].apply(apply_lemmatization)
reviews_df['replyContent'] = reviews_df['replyContent'].apply(apply_lemmatization)

# Print the first few rows to verify the changes
print(reviews_df.head())

# Create a stemmer object
stemmer = PorterStemmer()

# Function to apply stemming
def apply_stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Apply stemming to the 'content' and 'replyContent' columns
reviews_df['content'] = reviews_df['content'].apply(apply_stemming)
reviews_df['replyContent'] = reviews_df['replyContent'].apply(apply_stemming)

# Print the first few rows to verify the changes
print(reviews_df.head())
