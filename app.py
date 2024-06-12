from flask import Flask, request, render_template, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the models
model_files = ['Logistic Regression', 'XGBoosting', 'Voting Classifier', 'Stacking Classifier']
emotions = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
models = {file: {'model': joblib.load(f'{file}.pkl')} for file in model_files}

# Metrics for models
metrics = {'Logistic Regression':{'Accuracy': 85.03, 'Precision': 85.01, 'Recall': 85.03, 'F1_Score': 85.00}, 
           'XGBoosting':{'Accuracy': 86.36, 'Precision': 86.63, 'Recall': 86.36, 'F1_Score': 86.45}, 
           'Voting Classifier':{'Accuracy': 86.30, 'Precision': 86.33, 'Recall': 86.30, 'F1_Score': 86.30}, 
           'Stacking Classifier':{'Accuracy': 86.04, 'Precision': 86.56, 'Recall': 86.04, 'F1_Score': 86.17}}

# Adding metrics to models
for model in models.keys():
    models[model]['metrics'] = metrics[model]

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
    return render_template('index.html', models=models)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    input_vectorized = vectorizer.transform([input_text])
    
    predictions = {model_name: emotions[model['model'].predict(input_vectorized)[0]] for model_name, model in models.items()}
    prediction_colors = {model_name: emotion_colors[prediction] for model_name, prediction in predictions.items()}
    
    return render_template('index.html', input_text=input_text, predictions=predictions, prediction_colors=prediction_colors, models=models)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join("uploads", file.filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        
        data = pd.read_csv(filepath)
        results = data.copy()
        for model_name, model in models.items():
            predictions = model['model'].predict(vectorizer.transform(data['reviews']))
            results[model_name] = [emotions[pred] for pred in predictions]

        results.to_csv(filepath, index=False)
        
        emotion_counts = {model_name: results[model_name].value_counts(normalize=True) * 100 for model_name in models}
        
        # Prepare colors for each emotion
        emotion_colors_list = {model_name: [emotion_colors[emotion] for emotion in results[model_name].unique()] for model_name in models}
        return render_template('results.html', tables=results.to_html(classes='table table-striped table-bordered', index=False), filename=file.filename, emotion_counts=emotion_counts, emotion_colors_list=emotion_colors_list, models=models)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)