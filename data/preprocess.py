"""
Generic Data Preprocessing Pipeline
Supports:
- Classification
- Regression
- Mixed numerical + categorical features

Assumptions:
- Dataset is a CSV file
- Last column is the target variable
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -----------------------------
# Paths
# -----------------------------
RAW_DATA_PATH = "data/raw/dataset.csv"
PROCESSED_DATA_DIR = "data/processed"


# -----------------------------
# 1. Load Data
# -----------------------------
def load_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            "dataset.csv not found in data/raw/"
        )
    return pd.read_csv(RAW_DATA_PATH)


# -----------------------------
# 2. Basic Cleaning
# -----------------------------
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df


# -----------------------------
# 3. Split Features & Target
# -----------------------------
def split_features_target(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


# -----------------------------
# 4. Detect Problem Type
# -----------------------------
def detect_problem_type(y):
    """
    Heuristic:
    - Few unique values or object type → classification
    - Otherwise → regression
    """
    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    return "regression"


# -----------------------------
# 5. Build Feature Preprocessor
# -----------------------------
def build_feature_preprocessor(X):
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


# -----------------------------
# 6. Encode Target
# -----------------------------
def process_target(y, problem_type):
    if problem_type == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    else:
        y = y.values.astype("float32")

    return y


# -----------------------------
# 7. Train / Val / Test Split
# -----------------------------
def split_data(X, y, problem_type):
    stratify = y if problem_type == "classification" else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=stratify
    )

    stratify_temp = y_temp if problem_type == "classification" else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=stratify_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------
# 8. Save Processed Data
# -----------------------------
def save_arrays(data):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    names = [
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test"
    ]

    for name, array in zip(names, data):
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{name}.npy"), array)


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting features and target...")
    X, y = split_features_target(df)

    problem_type = detect_problem_type(y)
    print(f"Detected problem type: {problem_type}")

    print("Building preprocessing pipeline...")
    feature_preprocessor = build_feature_preprocessor(X)

    print("Processing target...")
    y = process_target(y, problem_type)

    print("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, problem_type
    )

    print("Applying feature preprocessing...")
    X_train = feature_preprocessor.fit_transform(X_train)
    X_val = feature_preprocessor.transform(X_val)
    X_test = feature_preprocessor.transform(X_test)

    print("Saving processed data...")
    save_arrays([
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ])

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
