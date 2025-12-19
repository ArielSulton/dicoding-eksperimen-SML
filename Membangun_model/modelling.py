"""
SMS Spam Classification Model Training
Author: Mochammad Ariel Sulton (arielsulton)
Description: Train spam classification model with MLflow tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# DagsHub Integration with error handling
try:
    import dagshub
    dagshub.init(repo_owner='arielsulton', repo_name='sms-spam-mlops', mlflow=True)
    print("✅ DagsHub connected successfully!")
except Exception as e:
    print(f"⚠️  DagsHub connection warning: {e}")
    print("Continuing with local MLflow tracking...")
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")


def load_preprocessed_data(file_path):
    """
    Load preprocessed SMS spam dataset
    
    Args:
        file_path (str): Path to preprocessed CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading preprocessed data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully! Shape: {df.shape}")
    return df


def prepare_features(df, max_features=5000):
    """
    Prepare features using TF-IDF vectorization
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        max_features (int): Maximum number of features for TF-IDF
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, vectorizer
    """
    print("\nPreparing features with TF-IDF vectorization...")
    
    # Extract features and labels
    X = df['message_clean']
    y = df['label_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_naive_bayes(X_train, y_train):
    """
    Train Multinomial Naive Bayes model
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        model: Trained Naive Bayes model
    """
    print("\nTraining Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        model: Trained Logistic Regression model
    """
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        model: Trained Random Forest model
    """
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Display results
    print(f"\n{model_name} Performance:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return metrics, cm


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def main():
    """
    Main training function
    """
    print("="*60)
    print("SMS SPAM CLASSIFICATION MODEL TRAINING")
    print("Author: Mochammad Ariel Sulton (arielsulton)")
    print("Connected to DagsHub MLflow Tracking")
    print("="*60)
    
    # DagsHub MLflow is already configured via dagshub.init()
    mlflow.set_experiment("SMS Spam Classification")
    
    # Load data
    data_path = "sms_spam_preprocessing.csv"
    df = load_preprocessed_data(data_path)
    
    # Prepare features
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(df)
    
    # Define models to train
    models = {
        'Naive Bayes': train_naive_bayes,
        'Logistic Regression': train_logistic_regression,
        'Random Forest': train_random_forest
    }
    
    # Create directory for artifacts
    os.makedirs('screenshots', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    # Train and evaluate each model
    results = {}
    
    for model_name, train_func in models.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        # Start MLflow run with autolog
        with mlflow.start_run(run_name=model_name):
            # Enable autologging
            mlflow.sklearn.autolog()
            
            # Train model
            model = train_func(X_train, y_train)
            
            # Evaluate model
            metrics, cm = evaluate_model(model, X_test, y_test, model_name)
            
            # Plot and save confusion matrix
            cm_path = f"screenshots/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
            plot_confusion_matrix(cm, model_name, cm_path)
            
            # Log confusion matrix as artifact
            mlflow.log_artifact(cm_path)
            
            # Store results
            results[model_name] = metrics
            
            print(f"\nMLflow run completed for {model_name}")
    
    # Display comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    print(comparison_df)
    
    # Save comparison
    comparison_df.to_csv('artifacts/model_comparison.csv')
    print(f"\nModel comparison saved to artifacts/model_comparison.csv")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
