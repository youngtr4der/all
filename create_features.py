import pandas as pd
import numpy as np

# --- Configuration ---
LOOKBACK_WINDOWS = [8, 24, 48, 72]  # (in hours)

def calculate_features(main_df: pd.DataFrame, derivatives_df: pd.DataFrame, lookback_windows: list) -> pd.DataFrame:
    """
    Calculates derivative-based features for multiple time windows and merges them into the main dataframe.

    Args:
        main_df: The main dataframe with spot price data.
        derivatives_df: The dataframe with derivatives data.
        lookback_windows: A list of integers representing the lookback periods in hours.

    Returns:
        The main dataframe with the new features merged in.
    """
    # --- 1. Data Preparation ---
    for df in [main_df, derivatives_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

    df = pd.merge(main_df, derivatives_df, on='timestamp', how='inner', suffixes=('_spot', '_deriv'))

    if 'volume_deriv' in df.columns:
        df = df.rename(columns={'volume_deriv': 'volume_derivatives'})

    # --- 2. Feature Calculation ---

    # Feature: oi_trend_confirmation_score (6-hour window, independent of the main loop)
    price_change_6h = df['close'].diff(6)
    volume_change_6h = df['volume_derivatives'].diff(6)
    oi_change_6h = df['open_interest'].diff(6)
    conditions = [
        (price_change_6h > 0) & (volume_change_6h > 0) & (oi_change_6h > 0),
        (price_change_6h < 0) & (volume_change_6h > 0) & (oi_change_6h > 0),
        (price_change_6h < 0) & (volume_change_6h < 0) & (oi_change_6h < 0),
        (price_change_6h > 0) & (volume_change_6h < 0) & (oi_change_6h < 0)
    ]
    choices = [2, -2, 1, -1]
    df['oi_trend_confirmation_score'] = np.select(conditions, choices, default=0)

    # --- Loop through specified lookback windows to generate features ---
    for window in lookback_windows:
        # Column name suffix for the current window
        suffix = f"_{window}h"

        # OI Growth Rate
        df[f'oi_growth_rate{suffix}'] = (df['open_interest'] / df['open_interest'].shift(window)) - 1

        # OI to Volume Ratio
        rolling_volume = df['volume_derivatives'].rolling(window=window).sum()
        df[f'oi_to_volume_ratio{suffix}'] = df['open_interest'] / rolling_volume

        # Funding Rate Moving Average
        df[f'funding_rate_ma{suffix}'] = df['funding_rate'].rolling(window=window).mean()

        # Funding Rate Z-score
        funding_mean = df[f'funding_rate_ma{suffix}'] # Use the MA we just calculated
        funding_std = df['funding_rate'].rolling(window=window).std()
        df[f'funding_rate_zscore{suffix}'] = (df['funding_rate'] - funding_mean) / funding_std
        df[f'funding_rate_zscore_abs{suffix}'] = df[f'funding_rate_zscore{suffix}'].abs()

        # OI Weighted Funding Rate
        df[f'oi_weighted_funding_rate{suffix}'] = df[f'funding_rate_ma{suffix}'] * df[f'oi_growth_rate{suffix}']

        # Funding-Spot Divergence
        spot_roc = (df['close'] / df['close'].shift(window)) - 1
        spot_roc_mean = spot_roc.rolling(window=window).mean()
        spot_roc_std = spot_roc.rolling(window=window).std()
        spot_roc_zscore = (spot_roc - spot_roc_mean) / spot_roc_std
        df[f'funding_spot_divergence{suffix}'] = df[f'funding_rate_zscore{suffix}'] - spot_roc_zscore

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Feature calculation complete for windows: {lookback_windows}.")
    return df


def main():
    """
    Main execution function to load data, generate features, and save the result.
    """
    MAIN_DATASET_PATH = 'BTC_USDT_1h_labeled_H24.csv'
    DERIVATIVES_PATH = 'derivatives_data.csv'
    OUTPUT_PATH = 'features_dataset_v2_multi_window.csv'

    print(f"Loading main dataset from '{MAIN_DATASET_PATH}'...")
    try:
        main_df = pd.read_csv(MAIN_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Main dataset not found at '{MAIN_DATASET_PATH}'.")
        return

    print(f"Loading derivatives dataset from '{DERIVATIVES_PATH}'...")
    try:
        derivatives_df = pd.read_csv(DERIVATIVES_PATH)
    except FileNotFoundError:
        print(f"Error: Derivatives dataset not found at '{DERIVATIVES_PATH}'.")
        return

    print("Calculating new features...")
    enriched_df = calculate_features(main_df, derivatives_df, LOOKBACK_WINDOWS)

    enriched_df.reset_index(inplace=True)
    print(f"Saving enriched dataset to '{OUTPUT_PATH}'...")
    try:
        enriched_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Successfully saved to '{OUTPUT_PATH}'.")
    except Exception as e:
        print(f"An error occurred while saving the final file: {e}")


if __name__ == "__main__":
    main()
