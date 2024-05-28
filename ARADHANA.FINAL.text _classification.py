#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load and Preprocess the Data


# In[2]:


get_ipython().system('pip install pandas nltk scikit-learn matplotlib seaborn imbalanced-learn')


# In[3]:


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
df = pd.read_csv('Emotions_training.csv')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lower case
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove links
    text = text.replace('\n', ' ')  # Remove new lines
    text = ' '.join([word for word in text.split() if not any(c.isdigit() for c in word)])  # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])  # Stemming
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
    return text

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)
print(df['text'].head())


# # 2. Feature Engineering
# Convert the text corpus to a matrix of word counts using TF-IDF, visualize the word counts, balance the data,
# and save the dataset in CSV format:

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Convert the text corpus to a matrix of word counts using TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# Visualize word counts
word_count_df = pd.DataFrame(X, columns=tfidf.get_feature_names_out())
word_count_df.sum().sort_values(ascending=False).head(20).plot(kind='bar', figsize=(12, 6))
plt.title('Top 20 Words by TF-IDF Score')
plt.show()


# In[5]:


#Balancing the dataset


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Convert the text corpus to a matrix of word counts using TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Extract the indices of the resampled data
resampled_indices = smote.fit_resample(X, y)[1]

# Extract the corresponding text data
texts_balanced = df.iloc[resampled_indices]['text'].values

# Combine text with TF-IDF features and labels
balanced_df = pd.DataFrame(X_balanced, columns=tfidf.get_feature_names_out())
balanced_df['text'] = texts_balanced
balanced_df['label'] = y_balanced

# Save the balanced dataset to CSV
balanced_df.to_csv('balanced_text_data.csv', index=False)

# Visualize the balanced data
sns.countplot(y_balanced)
plt.title('Balanced Class Distribution')
plt.show()


# In[7]:


#Split the Dataset: Divide the dataset into training (70%) and testing (20%) sets.


# In[8]:


from sklearn.model_selection import train_test_split

# Split data into training (70%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)


# # Text Classification 
# Modeling Approach

# # Model 1: Logistic Regression

# # Logistic Regression
# 
# Definition: Logistic Regression is a linear model used for binary classification tasks, where the output is a probability score that the input belongs to a certain class.
# 
# Model Type: Supervised learning model.
# 
# Output: Probability score between 0 and 1.
# 
# Decision Rule: Predicts the class with a threshold (typically 0.5).
# 
# Advantages:
# 
# Simple and easy to implement.
# Outputs can be interpreted as probabilities.
# Efficient to train and works well with large datasets.
# 
# Disadvantages:
# 
# Assumes linear decision boundaries.
# Limited expressiveness compared to more complex models

# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Train the model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# Generate classification report and confusion matrix
print("Logistic Regression - Training set report:")
print(classification_report(y_train, y_pred_train))
print("Logistic Regression - Test set report:")
print(classification_report(y_test, y_pred_test))

ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test)
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


# # Hyperparameter Tuning
# Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model. Hyperparameters are settings that are external to the model and whose values cannot be directly estimated from the data. They are typically set before the learning process begins.

# # Why Tune Hyperparameters?
# Improve Model Performance: Optimal hyperparameters can significantly improve model performance metrics like accuracy, precision, and recall.
# Generalization: Finding the best hyperparameters can help the model generalize better to new, unseen data.
# Avoid Overfitting: Proper hyperparameter tuning can prevent the model from overfitting or underfitting the training data.

# # Techniques for Hyperparameter Tuning
# Manual Search:
# 
# Manually select hyperparameters based on intuition and trial and error.
# Suitable for small datasets or when you have prior knowledge of the problem.
# Grid Search:
# 
# Define a grid of hyperparameter values.
# Evaluate each combination of values using cross-validation and select the best one.
# Suitable when the search space is not too large.
# Random Search:
# 
# Randomly sample combinations of hyperparameter values.
# Evaluate each combination using cross-validation.
# Suitable when the search space is large.
# Bayesian Optimization:
# 
# Uses probabilistic models to predict the performance of different hyperparameter configurations.
# Efficiently narrows down the search space based on past evaluations.
# Automated Hyperparameter Tuning:
# 
# Tools like scikit-learn's GridSearchCV and RandomizedSearchCV automate the process of grid search and random search, respectively.
# Libraries like Optuna, Hyperopt, and Ray Tune implement more advanced techniques.

# In[10]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_lr.fit(X_train, y_train)

# Best model parameters
print("Best parameters for Logistic Regression:", grid_lr.best_params_)

# Retrain with best parameters
best_lr = grid_lr.best_estimator_
y_pred_train = best_lr.predict(X_train)
y_pred_test = best_lr.predict(X_test)

# Generate classification report with tuned model
print("Logistic Regression (tuned) - Training set report:")
print(classification_report(y_train, y_pred_train))
print("Logistic Regression (tuned) - Test set report:")
print(classification_report(y_test, y_pred_test))

ConfusionMatrixDisplay.from_estimator(best_lr, X_test, y_test)
plt.title('Tuned Confusion Matrix for Logistic Regression')
plt.show()


# # Model 2: Random Forest

# # Random Forest
# 
# Definition: Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes as the prediction.
# 
# Model Type: Ensemble learning model (specifically, bagging).
# 
# Output: Mode of the classes (for classification tasks).
# 
# Decision Rule: Averages predictions over multiple decision trees.
# 
# Advantages:
# 
# High accuracy and robustness against overfitting.
# Handles both categorical and numerical data.
# Provides feature importance ranking.
# Disadvantages:
# 
# Can be slow to evaluate.
# More complex and harder to interpret than single decision trees.

# In[11]:


from sklearn.ensemble import RandomForestClassifier

# Train the model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Generate classification report and confusion matrix
print("Random Forest - Training set report:")
print(classification_report(y_train, y_pred_train))
print("Random Forest - Test set report:")
print(classification_report(y_test, y_pred_test))

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title('Confusion Matrix for Random Forest')
plt.show()

# Hyperparameter tuning
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_rf.fit(X_train, y_train)

# Best model parameters
print("Best parameters for Random Forest:", grid_rf.best_params_)

# Retrain with best parameters
best_rf = grid_rf.best_estimator_
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

# Generate classification report with tuned model
print("Random Forest (tuned) - Training set report:")
print(classification_report(y_train, y_pred_train))
print("Random Forest (tuned) - Test set report:")
print(classification_report(y_test, y_pred_test))

ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test)
plt.title('Tuned Confusion Matrix for Random Forest')
plt.show()


# # Model 3: Naive Bayes Classifier: 
# #Naive Bayes is a family of probabilistic classifiers based on Bayes' Theorem with the "naive" 
# #assumption of conditional independence between every pair of features given the value of the class variable.
# #It is particularly well-suited for text classification tasks, such as spam detection, 
# #sentiment analysis, and document categorization.

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Split data into training (70%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Train the Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_train = nb.predict(X_train)
y_pred_test = nb.predict(X_test)

# Generate classification report and confusion matrix
print("Naive Bayes - Training set report:")
print(classification_report(y_train, y_pred_train))
print("Naive Bayes - Test set report:")
print(classification_report(y_test, y_pred_test))

ConfusionMatrixDisplay.from_estimator(nb, X_test, y_test)
plt.title('Confusion Matrix for Naive Bayes')
plt.show()


# In[13]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


Hyperparameter Tuning


# In[14]:


# Define the parameter grid for alpha
param_grid = {'alpha': [0.01, 0.1, 1.0]}

# Initialize the Multinomial Naive Bayes model
nb = MultinomialNB()

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# Get the best model
best_nb = grid_search.best_estimator_

# Evaluate the best model
y_pred_train = best_nb.predict(X_train)
y_pred_test = best_nb.predict(X_test)

print("Naive Bayes (with best hyperparameters) - Training set report:")
print(classification_report(y_train, y_pred_train))
print("Naive Bayes (with best hyperparameters) - Test set report:")
print(classification_report(y_test, y_pred_test))

ConfusionMatrixDisplay.from_estimator(best_nb, X_test, y_test)
plt.title('Confusion Matrix for Naive Bayes (Best Hyperparameters)')
plt.show()


# # Final Model Selection
# Evaluate the final model on the validation set and choose the best performing one based on the classification report.

# In[ ]:


#Here we are taking Logistic Regression has our final model,Now we will apply the validation 10% to get final result.


# In[ ]:


# Split data into training (70%), testing (20%), and validation (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Define the parameter grid for Logistic Regression
param_grid = {'C': [0.1, 1, 10]}

# Initialize the Logistic Regression model
lr = LogisticRegression(max_iter=1000, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# Get the best model
best_lr = grid_search.best_estimator_

# Evaluate the best model on training, test, and validation sets
y_pred_train = best_lr.predict(X_train)
y_pred_test = best_lr.predict(X_test)
y_pred_val = best_lr.predict(X_val)

print("Logistic Regression (with best hyperparameters) - Training set report:")
print(classification_report(y_train, y_pred_train))
print("Logistic Regression (with best hyperparameters) - Test set report:")
print(classification_report(y_test, y_pred_test))
print("Logistic Regression (with best hyperparameters) - Validation set report:")
print(classification_report(y_val, y_pred_val))

ConfusionMatrixDisplay.from_estimator(best_lr, X_test, y_test)
plt.title('Confusion Matrix for Logistic Regression (Best Hyperparameters)')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




