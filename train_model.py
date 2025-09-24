import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate():
    """
    This script performs a full cycle of training and evaluating a classification model.
    """
    # 1. Load Data
    try:
        df = pd.read_csv('features_dataset_v1.csv')
    except FileNotFoundError:
        print("Error: 'features_dataset_v1.csv' not found. Please run create_features.py first.")
        return

    # 2. Define X and y
    # The target variable is 'y_target'.
    y = df['y_target']

    # Features are the 11 columns from 'open_interest' to 'funding_spot_divergence'.
    # This is based on the analysis of create_features.py, as the names provided in the
    # prompt (e.g., 'voi_level1') were not found in the dataset.
    feature_columns = [
        'open_interest', 'funding_rate', 'volume_derivatives', 'oi_growth_rate_24h',
        'oi_trend_confirmation_score', 'oi_to_volume_ratio_24h', 'funding_rate_ma_8h',
        'funding_rate_zscore_24h', 'funding_rate_zscore_abs', 'oi_weighted_funding_rate',
        'funding_spot_divergence'
    ]
    X = df[feature_columns]

    # Convert timestamp column to datetime for splitting
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 3. Temporal Split
    # Training data: everything before 2024-01-01
    # Test data: everything on and after 2021-02-01
    # The original split date '2024-01-01' resulted in an empty test set,
    # as the data only goes up to 2021-02-11.
    split_date = pd.to_datetime('2021-02-01')

    train_mask = df['timestamp'] < split_date
    test_mask = df['timestamp'] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Handle potential NaN values that may have resulted from feature calculations (e.g., rolling windows)
    # For this baseline model, we will fill them with 0.
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print("-" * 30)

    # 4. Train Model
    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Training complete.")
    print("-" * 30)

    # 5. Evaluate Model
    print("Evaluating model on the test set...")
    y_pred = model.predict(X_test)

    # Print Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 30)

    # Print Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate()
