import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')


df = pd.read_csv('DataPreProcessed.csv')


df = df.dropna(subset=['content'])


sid = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    scores = sid.polarity_scores(' '.join(filtered_tokens))
    return scores


df['sentiment_scores'] = df['content'].apply(lambda x: analyze_sentiment(str(x)))


print(df['sentiment_scores'])
df.to_csv('sentiment_scores.csv', index=False)

