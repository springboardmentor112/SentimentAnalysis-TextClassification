import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load the CSV file
df = pd.read_csv('DataPreProcessed.csv')

# Drop rows with NaN values in the 'content' column
df = df.dropna(subset=['content'])

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Function to tokenize text and calculate sentiment score
def analyze_sentiment(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    scores = sid.polarity_scores(' '.join(filtered_tokens))
    return scores

# Apply sentiment analysis to the 'content' column
df['sentiment_scores'] = df['content'].apply(lambda x: analyze_sentiment(str(x)))

# Print the sentiment scores
print(df['sentiment_scores'])
df.to_csv('sentiment_scores.csv', index=False)

