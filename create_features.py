import pandas as pd
import numpy as np

def calculate_features(main_df: pd.DataFrame, derivatives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 8 derivative-based features and merges them into the main dataframe.

    Args:
        main_df: The main dataframe with spot price data (must include 'timestamp' and 'close').
        derivatives_df: The dataframe with derivatives data (must include 'timestamp',
                        'open_interest', 'funding_rate', 'volume').

    Returns:
        The main dataframe with the 8 new features merged in.
    """
    # --- 1. Data Preparation ---
    # Ensure 'timestamp' is the index and is in datetime format for both dataframes
    for df in [main_df, derivatives_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

    # Merge the two dataframes to align all data. We only keep timestamps that exist in both.
    # We also need the 'close' price from the main_df for some calculations.
    df = pd.merge(main_df, derivatives_df, on='timestamp', how='inner', suffixes=('_spot', '_deriv'))

    # For clarity, let's rename the derivatives volume column if it exists
    if 'volume_deriv' in df.columns:
        df = df.rename(columns={'volume_deriv': 'volume_derivatives'})

    # --- 2. Feature Calculation ---

    # Feature 7: oi_growth_rate_24h (24-hour open interest growth rate)
    # Logic: Percentage change in Open Interest (OI) over the last 24 hours.
    df['oi_growth_rate_24h'] = (df['open_interest'] / df['open_interest'].shift(24)) - 1

    # Feature 8: oi_trend_confirmation_score (OI Trend Confirmation Score)
    # Logic: A composite score to evaluate trend strength over a 6-hour period.
    # We calculate 6-hour changes for price, volume, and open interest.
    price_change_6h = df['close'].diff(6)
    volume_change_6h = df['volume_derivatives'].diff(6)
    oi_change_6h = df['open_interest'].diff(6)

    conditions = [
        (price_change_6h > 0) & (volume_change_6h > 0) & (oi_change_6h > 0),  # Price, Vol, OI up
        (price_change_6h < 0) & (volume_change_6h > 0) & (oi_change_6h > 0),  # Price down, Vol, OI up
        (price_change_6h < 0) & (volume_change_6h < 0) & (oi_change_6h < 0),  # Price, Vol, OI down
        (price_change_6h > 0) & (volume_change_6h < 0) & (oi_change_6h < 0)   # Price up, Vol, OI down
    ]
    choices = [2, -2, 1, -1]
    df['oi_trend_confirmation_score'] = np.select(conditions, choices, default=0)

    # Feature 9: oi_to_volume_ratio_24h (24-hour OI to Volume Ratio)
    # Logic: Ratio of current OI to the sum of trading volume over the last 24 hours.
    rolling_volume_24h = df['volume_derivatives'].rolling(window=24).sum()
    df['oi_to_volume_ratio_24h'] = df['open_interest'] / rolling_volume_24h

    # Feature 10: funding_rate_ma_8h (8-hour moving average of funding rate)
    # Logic: Simple moving average of the funding rate over 8 hours.
    df['funding_rate_ma_8h'] = df['funding_rate'].rolling(window=8).mean()

    # Feature 11: funding_rate_zscore_24h (24-hour Z-score of funding rate)
    # Logic: How many standard deviations the current funding rate is from its 24-hour mean.
    funding_mean_24h = df['funding_rate'].rolling(window=24).mean()
    funding_std_24h = df['funding_rate'].rolling(window=24).std()
    df['funding_rate_zscore_24h'] = (df['funding_rate'] - funding_mean_24h) / funding_std_24h

    # Feature 12: funding_rate_zscore_abs (Absolute value of Z-score)
    # Logic: The absolute value of Feature 11, showing anomaly magnitude.
    df['funding_rate_zscore_abs'] = df['funding_rate_zscore_24h'].abs()

    # Feature 13: oi_weighted_funding_rate (Funding Rate weighted by OI Growth)
    # Logic: Product of the smoothed funding rate and the OI growth rate.
    df['oi_weighted_funding_rate'] = df['funding_rate_ma_8h'] * df['oi_growth_rate_24h']

    # Feature 14: funding_spot_divergence (Divergence between Funding and Spot)
    # Logic: Difference between the Z-score of funding rate and the Z-score of spot price's rate of change.
    # Step 14a: Calculate 24h rate of change for spot price.
    spot_roc_24h = (df['close'] / df['close'].shift(24)) - 1
    # Step 14b: Calculate Z-score for the spot roc.
    spot_roc_mean_24h = spot_roc_24h.rolling(window=24).mean()
    spot_roc_std_24h = spot_roc_24h.rolling(window=24).std()
    spot_roc_24h_zscore = (spot_roc_24h - spot_roc_mean_24h) / spot_roc_std_24h
    # Step 14c: Calculate the final divergence feature.
    df['funding_spot_divergence'] = df['funding_rate_zscore_24h'] - spot_roc_24h_zscore

    # Replace infinite values that can occur from division by zero with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Feature calculation complete.")
    return df


def main():
    """
    Main execution function to load data, generate features, and save the result.
    """
    # --- Configuration ---
    MAIN_DATASET_PATH = 'BTC_USDT_1h_labeled_H24.csv'
    DERIVATIVES_PATH = 'derivatives_data.csv'
    OUTPUT_PATH = 'features_dataset_v1.csv'

    # --- Load Data ---
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

    # --- Calculate Features ---
    print("Calculating new features...")
    enriched_df = calculate_features(main_df, derivatives_df)

    # --- Save Final DataFrame ---
    # We reset the index to bring 'timestamp' back as a column
    enriched_df.reset_index(inplace=True)

    print(f"Saving enriched dataset to '{OUTPUT_PATH}'...")
    try:
        enriched_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Successfully saved to '{OUTPUT_PATH}'.")
    except Exception as e:
        print(f"An error occurred while saving the final file: {e}")


if __name__ == "__main__":
    main()
