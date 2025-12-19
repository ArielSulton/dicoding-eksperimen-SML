"""
SMS Spam Classification Model Training with Hyperparameter Tuning
Author: Mochammad Ariel Sulton (arielsulton)
Description: Train spam classification model with hyperparameter tuning and manual MLflow logging
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json

USE_DAGSHUB = True

if USE_DAGSHUB:
    try:
        # Try Method 1: dagshub.init
        import dagshub
        dagshub.init(repo_owner='arielsulton', repo_name='sms-spam-mlops', mlflow=True)
        print("✅ DagsHub connected via dagshub.init!")
    except Exception as e:
        print(f"⚠️  DagsHub init failed: {str(e)[:100]}")
        print("Falling back to local MLflow...")
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        USE_DAGSHUB = False  # Fallback to local
else:
    # Local MLflow
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    print("🔧 Using local MLflow tracking at http://127.0.0.1:5000/")


def load_preprocessed_data(file_path):
    """Load preprocessed SMS spam dataset"""
    print(f"Loading preprocessed data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully! Shape: {df.shape}")
    return df


def prepare_features(df, max_features=5000):
    """Prepare features using TF-IDF vectorization"""
    print("\nPreparing features with TF-IDF vectorization...")
    
    # Clean data - remove NaN values
    df = df.dropna(subset=['message_clean', 'label_encoded'])
    print(f"Data after cleaning: {len(df)} samples")
    
    X = df['message_clean']
    y = df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def tune_naive_bayes(X_train, y_train):
    """
    Tune Multinomial Naive Bayes with hyperparameter search
    """
    print("\nTuning Multinomial Naive Bayes...")
    
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'fit_prior': [True, False]
    }
    
    model = MultinomialNB()
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_logistic_regression(X_train, y_train):
    """
    Tune Logistic Regression with hyperparameter search
    """
    print("\nTuning Logistic Regression...")
    
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    }
    
    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest with hyperparameter search
    """
    print("\nTuning Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance with comprehensive metrics"""
    print(f"\nEvaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Performance:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return metrics, cm, y_pred, y_pred_proba


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot and save confusion matrix"""
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


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
    """Plot and save ROC curve"""
    if y_pred_proba is None:
        return
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_metrics_comparison(results_df, save_path=None):
    """Plot metrics comparison across models"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    results_df[['accuracy', 'precision', 'recall', 'f1_score']].plot(
        kind='bar', ax=ax, width=0.8
    )
    
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.legend(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    plt.close()


def main():
    """Main training function with hyperparameter tuning"""
    print("="*60)
    print("SMS SPAM CLASSIFICATION - HYPERPARAMETER TUNING")
    print("Author: Mochammad Ariel Sulton (arielsulton)")
    if USE_DAGSHUB:
        print("Tracking: DagsHub MLflow")
    else:
        print("Tracking: Local MLflow (http://127.0.0.1:5000/)")
    print("="*60)
    
    # MLflow experiment
    mlflow.set_experiment("SMS Spam Classification - Tuned")
    
    # Load data
    data_path = "sms_spam_preprocessing.csv"
    df = load_preprocessed_data(data_path)
    
    # Prepare features
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(df)
    
    # Create directories
    os.makedirs('screenshots', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    # Define tuning functions
    tuning_functions = {
        'Naive Bayes (Tuned)': tune_naive_bayes,
        'Logistic Regression (Tuned)': tune_logistic_regression,
        'Random Forest (Tuned)': tune_random_forest
    }
    
    # Train and evaluate each model
    results = {}
    
    for model_name, tune_func in tuning_functions.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=model_name):
            # Disable autolog for manual logging
            mlflow.sklearn.autolog(disable=True)
            
            # Tune model
            model, best_params, best_cv_score = tune_func(X_train, y_train)
            
            # Log hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_f1_score", best_cv_score)
            
            # Evaluate model
            metrics, cm, y_pred, y_pred_proba = evaluate_model(
                model, X_test, y_test, model_name
            )
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Plot and log confusion matrix
            cm_path = f"screenshots/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
            plot_confusion_matrix(cm, model_name, cm_path)
            mlflow.log_artifact(cm_path)
            
            # Plot and log ROC curve
            if y_pred_proba is not None:
                roc_path = f"screenshots/roc_curve_{model_name.replace(' ', '_').lower()}.png"
                plot_roc_curve(y_test, y_pred_proba, model_name, roc_path)
                mlflow.log_artifact(roc_path)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log vectorizer
            import pickle
            vectorizer_path = f"artifacts/vectorizer_{model_name.replace(' ', '_').lower()}.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            mlflow.log_artifact(vectorizer_path)
            
            # Store results
            results[model_name] = metrics
            
            print(f"\nMLflow run completed for {model_name}")
    
    # Display comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON (TUNED)")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    print(comparison_df)
    
    # Save comparison
    comparison_path = 'artifacts/model_comparison_tuned.csv'
    comparison_df.to_csv(comparison_path)
    
    # Plot metrics comparison
    comparison_plot_path = 'screenshots/model_comparison_tuned.png'
    plot_metrics_comparison(comparison_df, comparison_plot_path)
    
    print(f"\nModel comparison saved to {comparison_path}")
    print(f"Comparison plot saved to {comparison_plot_path}")
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
