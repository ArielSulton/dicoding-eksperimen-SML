"""
SMS Spam Classification - Basic MLflow Autolog
Author: Mochammad Ariel Sulton (arielsulton)
Description: Basic model training with mlflow.autolog() only
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Setup DagsHub
try:
    import dagshub
    dagshub.init(repo_owner='arielsulton', repo_name='sms-spam-mlops', mlflow=True)
    print("✅ DagsHub connected")
except Exception as e:
    print(f"⚠️ DagsHub warning: {e}")
    print("Continuing with local MLflow...")

def main():
    # Set experiment
    mlflow.set_experiment("SMS Spam Classification - Basic Autolog")
    
    # Enable autolog - automatic logging
    mlflow.autolog()
    
    print("Loading data from sms_spam_preprocessing.csv...")
    df = pd.read_csv('sms_spam_preprocessing.csv')
    
    # Clean NaN values
    df = df.dropna(subset=['message_clean', 'label_encoded'])
    print(f"Data loaded: {len(df)} samples")
    
    X = df['message_clean']
    y = df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Preparing features with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    with mlflow.start_run(run_name="logistic_regression_autolog"):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_tfidf, y_train)
        
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained! Accuracy: {accuracy:.4f}")
        print("✅ All metrics logged automatically by autolog")

if __name__ == "__main__":
    main()
