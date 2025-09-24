import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings

# Suppress FutureWarning from hmmlearn
warnings.filterwarnings("ignore", category=FutureWarning)

def prepare_data_for_hmm(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Prepares raw OHLCV data for HMM training.

    Args:
        df: DataFrame with at least 'timestamp' and 'close' columns.

    Returns:
        A tuple containing:
        - The prepared DataFrame with a datetime index and 'log_return' column.
        - A DataFrame with features for the HMM model.
    """
    # Ensure timestamp is a column if it's the index
    if 'timestamp' not in df.columns:
        df = df.reset_index()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df.sort_index(inplace=True)

    df['log_return'] = np.log(df['close']).diff()

    # Drop rows with NaN values that result from differencing
    df = df.dropna(subset=['log_return'])

    # The feature for the HMM is the log return
    hmm_features = df[['log_return']]

    return df, hmm_features

def get_standardized_state_mapping(hmm_model: GaussianHMM) -> dict:
    """
    Sorts HMM states based on their associated volatility (standard deviation)
    and returns a mapping from the original state label to a standardized one.

    State 0: Lowest volatility
    State 1: Medium volatility
    ...
    State N: Highest volatility

    Args:
        hmm_model: A trained GaussianHMM model.

    Returns:
        A dictionary mapping original state labels to standardized labels.
    """
    # Volatility is the standard deviation of the Gaussian emission of a state.
    # For a single-feature model, `covars_` is of shape (n_components, 1, 1).
    volatilities = np.sqrt(hmm_model.covars_[:, 0, 0])

    # Sort states by volatility: argsort returns the indices that would sort the array
    sorted_state_indices = np.argsort(volatilities)

    # Create the mapping: {original_state: standardized_state}
    # e.g., if state 2 has the lowest volatility, state 0 the medium, and state 1 the highest,
    # the mapping will be {2: 0, 0: 1, 1: 2}
    state_mapping = {original_idx: standardized_idx for standardized_idx, original_idx in enumerate(sorted_state_indices)}

    return state_mapping

def calculate_target_variable(df: pd.DataFrame, horizon: int, high_vol_regime_label: int) -> pd.DataFrame:
    """
    Calculates the binary target variable 'y_target'. 'y_target' is 1 if the
    high-volatility regime is observed within the next 'horizon' periods
    (from t+1 to t+horizon), and 0 otherwise.

    Args:
        df: DataFrame with a 'regime' column.
        horizon: The prediction horizon (number of periods to look ahead).
        high_vol_regime_label: The integer label for the high-volatility regime.

    Returns:
        The DataFrame with the new 'y_target' column.
    """
    # Create a boolean series: True where the regime is high-volatility
    is_high_vol = (df['regime'] == high_vol_regime_label)

    # Use rolling 'max' to see if a high-volatility event occurs in a window.
    # .rolling() looks backwards by default. To make it look forward, we shift
    # the result of the rolling operation. Shifting by -horizon means the value
    # at time `t` will be the result of the window from `t+1` to `t+horizon`.
    y_target_series = is_high_vol.rolling(window=horizon, min_periods=1).max().shift(-horizon)

    # The last `horizon` values will be NaN. We fill them with 0.
    df['y_target'] = y_target_series.fillna(0).astype(int)

    return df

def main():
    """
    Main execution function to run the entire data processing pipeline.
    """
    # --- Configuration ---
    H = 24  # Prediction horizon
    N_STATES = 3  # Number of volatility regimes (e.g., low, medium, high)
    INPUT_CSV = 'BTC_USDT_1h_from_2021-01-01.csv'
    OUTPUT_CSV = f'BTC_USDT_1h_labeled_H{H}.csv'

    # --- 1. Load Data ---
    print(f"1. Loading data from '{INPUT_CSV}'...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV}'.")
        return

    # --- 2. Prepare Data for HMM ---
    print("2. Preparing data: calculating log returns...")
    df_prepared, hmm_features = prepare_data_for_hmm(df)

    # --- 3. Train HMM Model ---
    print(f"3. Training GaussianHMM with {N_STATES} states...")
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full", # Each state has its own full covariance matrix
        n_iter=100,             # Number of iterations for the EM algorithm
        random_state=42         # For reproducibility
    )
    model.fit(hmm_features)
    print("   Model training complete.")

    # --- 4. Predict Regimes and Standardize ---
    print("4. Predicting volatility regimes...")
    hidden_states = model.predict(hmm_features)
    df_prepared['regime_original'] = hidden_states

    print("   Standardizing regime labels (0=low_vol, 1=medium_vol, 2=high_vol)...")
    state_mapping = get_standardized_state_mapping(model)
    df_prepared['regime'] = df_prepared['regime_original'].map(state_mapping)

    # The highest volatility regime will have the highest standardized label
    high_vol_label = N_STATES - 1

    # --- 5. Calculate Target Variable ---
    print(f"5. Calculating target variable 'y_target' for horizon H={H}...")
    df_final = calculate_target_variable(df_prepared, H, high_vol_label)

    # --- 6. Display Results ---
    print("\n--- Final DataFrame ---")
    print("Head:")
    print(df_final.head())
    print("\nTail:")
    # The tail will have y_target=0 for the last H rows
    print(df_final.tail())

    print("\n--- Class Distribution for 'y_target' ---")
    print(df_final['y_target'].value_counts())

    # --- 7. Save Final DataFrame ---
    print(f"\n7. Saving labeled data to '{OUTPUT_CSV}'...")
    df_final.to_csv(OUTPUT_CSV)
    print(f"   Successfully saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()
