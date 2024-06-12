from flask import Flask, request, render_template, redirect, url_for, send_file
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the models
model_files = ['Logistic Regression', 'XGBoosting', 'Voting Classifier', 'Stacking Classifier']
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

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('results', filename=file.filename))
    
    return redirect(request.url)

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    input_texts = df['reviews']
    input_vectorized = vectorizer.transform(input_texts)
    
    for model_name, model in models.items():
        df[model_name] = [emotions[model.predict(vectorizer.transform([text]))[0]] for text in input_texts]
    
    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
    df.to_csv(result_filepath, index=False)
    
    emotion_counts = {model_name: df[model_name].value_counts(normalize=True) * 100 for model_name in models.keys()}
    
    return render_template('results.html', tables=[df.head(10).to_html(classes='data', index=False)], titles=df.columns.values, emotion_counts=emotion_counts, filename=f'result_{filename}')

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)