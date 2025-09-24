import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import os

def generate_mock_dataset(filename="features_dataset_v2_multi_window.csv"):
    """
    Generates a mock dataset that mimics the structure of the user's dataset.
    This is necessary because the original CSV is not available.
    """
    print("Generating mock dataset...")
    # Create a date range
    timestamps = pd.to_datetime(pd.date_range(start='2021-01-01', end='2024-06-01', freq='h'))
    n = len(timestamps)

    # Create base data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.uniform(20000, 70000, n),
        'high': lambda x: x['open'] + np.random.uniform(0, 1000, n),
        'low': lambda x: x['open'] - np.random.uniform(0, 1000, n),
        'close': lambda x: x['open'] + np.random.uniform(-500, 500, n),
        'volume_spot': np.random.uniform(1000, 10000, n),
        'open_interest': np.random.uniform(40000, 60000, n),
        'funding_rate': np.random.normal(0.0001, 0.0005, n),
        'volume_derivatives': np.random.uniform(1000, 10000, n),
        'y_target': np.random.randint(0, 2, n)
    })

    # Calculate features for different windows
    windows = [8, 24, 48, 72]
    for w in windows:
        # Simple moving average for demonstration
        data[f'funding_rate_ma_{w}h'] = data['funding_rate'].rolling(window=w).mean()

        # Z-score
        data[f'funding_rate_zscore_{w}h'] = (data['funding_rate'] - data[f'funding_rate_ma_{w}h']) / data['funding_rate'].rolling(window=w).std()

        # OI Growth Rate
        data[f'oi_growth_rate_{w}h'] = data['open_interest'].pct_change(periods=w)

        # OI to Volume Ratio
        data[f'oi_to_volume_ratio_{w}h'] = data['open_interest'] / data['volume_derivatives'].rolling(window=w).sum()

        # Other features (simplified for mock generation)
        data[f'funding_rate_zscore_abs_{w}h'] = data[f'funding_rate_zscore_{w}h'].abs()
        data[f'oi_weighted_funding_rate_{w}h'] = data['funding_rate'] * data['open_interest']
        data[f'funding_spot_divergence_{w}h'] = data['funding_rate'] - data['volume_spot'].pct_change(w)


    # Fill NaNs that might have been generated at the start
    data = data.fillna(0)

    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Mock dataset saved to {filename}")
    return data

def analyze_and_train():
    """
    Main function to run the analysis and training pipeline.
    """
    DATASET_PATH = "features_dataset_v2_multi_window.csv"

    # --- 1. Data Generation and Loading ---
    if not os.path.exists(DATASET_PATH):
        print(f"{DATASET_PATH} not found. Generating a mock dataset.")
        generate_mock_dataset(DATASET_PATH)
    else:
        print(f"Loading existing dataset from {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Drop rows with NaN values that might persist
    df.dropna(inplace=True)

    print("Data loaded successfully. Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # --- 2. Analysis and Visualization ---
    print("\n--- Starting Analysis and Visualization ---")

    # Create a directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    features_to_analyze = [
        'oi_growth_rate_8h',
        'funding_rate_zscore_24h',
        'oi_to_volume_ratio_48h',
        'oi_growth_rate_72h',
        'funding_spot_divergence_48h'
    ]

    for feature in features_to_analyze:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in the dataframe. Skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x=feature, hue='y_target', fill=True, common_norm=False)
        plt.title(f'Density Plot of {feature} by y_target')
        plt.xlabel(feature)
        plt.ylabel('Density')

        plot_filename = f'plots/{feature}_density_plot.png'
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

    # --- 3. Model Training (v3) ---
    print("\n--- Starting Model Training ---")

    # Prepare data
    # Use all columns that are features (e.g., end with _8h, _24h, etc.)
    feature_cols = [col for col in df.columns if any(s in col for s in ['_8h', '_24h', '_48h', '_72h'])]

    X = df[feature_cols]
    y = df['y_target']

    print(f"Number of features: {len(X.columns)}")

    # Time-based split
    split_date = '2024-01-01'
    X_train = X.loc[X.index < split_date]
    y_train = y.loc[y.index < split_date]
    X_test = X.loc[X.index >= split_date]
    y_test = y.loc[y.index >= split_date]

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    # Initialize and train the model
    model = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=1000, # Increased estimators
        learning_rate=0.05,
        num_leaves=31
    )

    print("Training LGBMClassifier...")
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=True)])

    # --- 4. Model Evaluation ---
    print("\n--- Starting Model Evaluation ---")

    y_pred = model.predict(X_test)

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['y=0', 'y=1']))

    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix_v3.png')
    print("Saved confusion matrix plot to plots/confusion_matrix_v3.png")
    plt.show()


if __name__ == '__main__':
    analyze_and_train()
