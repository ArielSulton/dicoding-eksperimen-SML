"""
Automated SMS Spam Preprocessing Script
Author: Mochammad Ariel Sulton (arielsulton)
Description: Automates the preprocessing of SMS Spam dataset
"""

import pandas as pd
import re
import os
from pathlib import Path

def load_raw_data(file_path):
    """
    Load the SMS Spam Collection dataset
    
    Args:
        file_path (str): Path to the raw SMS Spam Collection file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    
    # Load the dataset with tab delimiter
    data = pd.read_csv(
        file_path,
        sep='\t',
        names=['label', 'message'],
        encoding='utf-8'
    )
    
    print(f"Data loaded successfully! Shape: {data.shape}")
    return data


def clean_text(text):
    """
    Clean and normalize text data
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d{5,}', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_data(df):
    """
    Preprocess the SMS Spam dataset
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    print("Starting preprocessing...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Check for missing values
    print(f"\nMissing values before cleaning:")
    print(df_clean.isnull().sum())
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"\nRemoved {initial_rows - len(df_clean)} duplicate rows")
    
    # Remove any rows with missing values
    df_clean = df_clean.dropna()
    print(f"Rows after removing missing values: {len(df_clean)}")
    
    # Clean the message text
    print("\nCleaning text messages...")
    df_clean['message_clean'] = df_clean['message'].apply(clean_text)
    
    # Add message length features
    df_clean['message_length'] = df_clean['message'].apply(len)
    df_clean['word_count'] = df_clean['message'].apply(lambda x: len(str(x).split()))
    
    # Encode labels (ham=0, spam=1)
    df_clean['label_encoded'] = df_clean['label'].map({'ham': 0, 'spam': 1})
    
    # Display class distribution
    print(f"\nClass distribution:")
    print(df_clean['label'].value_counts())
    print(f"\nClass distribution (%):")
    print(df_clean['label'].value_counts(normalize=True) * 100)
    
    print("\nPreprocessing completed!")
    
    return df_clean


def save_preprocessed_data(df, output_path):
    """
    Save preprocessed data to CSV file
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        output_path (str): Path to save the preprocessed data
    """
    print(f"\nSaving preprocessed data to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Data saved successfully!")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("SMS SPAM PREPROCESSING AUTOMATION")
    print("Author: Mochammad Ariel Sulton (arielsulton)")
    print("=" * 60)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    raw_data_path = base_dir / "sms_spam_raw" / "SMSSpamCollection"
    preprocessed_data_path = base_dir / "preprocessing" / "sms_spam_preprocessing.csv"
    
    # Load raw data
    df = load_raw_data(raw_data_path)
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Save preprocessed data
    save_preprocessed_data(df_clean, preprocessed_data_path)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
