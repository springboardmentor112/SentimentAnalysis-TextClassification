import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('reviews_sudheer_sir.csv')
# load the file using pandas
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
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Stemming
        porter = PorterStemmer()
        tokens = [porter.stem(word) for word in tokens]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into a string
        text = ' '.join(tokens)
    else:
        text = str(text)
    # Return preprocessed text
    return text


df['content'] = df['content'].apply(preprocess_text)
# Save preprocessed data to a new file
df.to_csv('preprocessed_file.csv', index=False)
