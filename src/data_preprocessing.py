# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    """Loads and preprocesses the Telco churn dataset."""
    df = pd.read_csv(filepath)

    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Handle blank TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)  # Drop rows where TotalCharges was blank

    # Encode binary Yes/No to 1/0
    binary_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Convert 'gender' to binary: Male = 1, Female = 0
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # SeniorCitizen is already 0/1 but might be int64
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    # One-hot encode multi-class categoricals
    multi_cat_cols = [
        'InternetService', 'Contract', 'PaymentMethod',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # Scale numeric columns
    scaler = StandardScaler()
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        df[col] = scaler.fit_transform(df[[col]])

    # Separate features and label
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # First split: train (80%) and temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: dev (10%) and test (10%) from temp
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )


    return X_train, X_dev, X_test, y_train, y_dev, y_test


if __name__ == "__main__":
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_and_clean_data("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn 2.csv")
    print("Train:", X_train.shape, "Dev:", X_dev.shape, "Test:", X_test.shape)
    print("Train churn ratio:", y_train.mean())
    print("Dev churn ratio:", y_dev.mean())
    print("Test churn ratio:", y_test.mean())

