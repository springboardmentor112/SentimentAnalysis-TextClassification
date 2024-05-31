from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the models
model_files = ['Logistic Regression', 
               'XGBoosting', 
               'Voting Classifier', 
               'Stacking Classifier']
emotions = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
models = {file: joblib.load(f'{file}.pkl') for file in model_files}

# Load the vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Color map for emotions
emotion_colors = {
    'Sadness': 'blue',
    'Joy': 'green',
    'Love': 'pink',
    'Anger': 'red',
    'Fear': 'purple',
    'Surprise': 'orange'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    input_vectorized = vectorizer.transform([input_text])
    
    predictions = {model_name: emotions[model.predict(input_vectorized)[0]] for model_name, model in models.items()}
    prediction_colors = {model_name: emotion_colors[prediction] for model_name, prediction in predictions.items()}
    
    return render_template('index.html', input_text=input_text, predictions=predictions, prediction_colors=prediction_colors)

if __name__ == '__main__':
    app.run(debug=True)
