import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load the CSV file
df = pd.read_csv('reviews_sudheer_sir.csv')

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove links
        text = re.sub(r'http\S+', '', text)
        # Remove next lines
        text = text.replace('\n', ' ')
        # Remove words containing numbers
        text = re.sub(r'\w*\d\w*', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        # Stemming
        porter = PorterStemmer()
        text = ' '.join([porter.stem(word) for word in text.split()])
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    else:
        return ""  # Return an empty string for NaN values

# Apply preprocessing to the content column
df['cleaned_reviews'] = df['content'].apply(preprocess_text)

# Define the sentiment analysis function with an expanded lexicon
def analyze_sentiment(tokens, lexicon):
    score = 0
    for word in tokens:
        if word in lexicon:
            score += lexicon[word]
    return score

# Load and expand lexicon for sentiment analysis
lexicon = {
    'good': 1,
    'bad': -1,
    'neutral': 0
    # Add more words and their sentiment scores as needed
}

# Tokenize the cleaned reviews
df['tokens'] = df['cleaned_reviews'].apply(word_tokenize)

# Apply sentiment analysis to cleaned reviews
df['sentiment_score'] = df['tokens'].apply(lambda x: analyze_sentiment(x, lexicon))

# Map sentiment scores to sentiment labels
df['review_sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Count the occurrences of each sentiment category
sentiment_counts = df['review_sentiment'].value_counts()

# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['red', 'grey', 'green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Print sentiment counts
print(sentiment_counts)
