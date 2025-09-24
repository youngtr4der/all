import pandas as pd
import numpy as np

# Load the main dataset to get the timestamps, which ensures alignment
try:
    main_df = pd.read_csv('BTC_USDT_1h_labeled_H24.csv')
    print(f"Loaded {len(main_df)} timestamps from BTC_USDT_1h_labeled_H24.csv")
except FileNotFoundError:
    print("Error: BTC_USDT_1h_labeled_H24.csv not found. Cannot create dummy data.")
    exit()

# Create a new DataFrame for the derivatives data
derivatives_df = pd.DataFrame()
derivatives_df['timestamp'] = main_df['timestamp']

# Get the number of rows to generate data for
num_rows = len(derivatives_df)

# Generate plausible random data for the required columns
# Open Interest: A large, slowly fluctuating number
oi_base = 50000
oi_trend = np.linspace(0, 10000, num_rows)
oi_noise = np.random.normal(0, 2000, num_rows)
derivatives_df['open_interest'] = oi_base + oi_trend + oi_noise

# Funding Rate: A small number, typically fluctuating around zero
derivatives_df['funding_rate'] = np.random.normal(0.0001, 0.0003, size=num_rows)

# Volume: Let's make it related to the spot volume for some realism
# but not identical. We'll use a random factor.
# If 'volume' isn't in main_df, just create random data.
if 'volume' in main_df.columns:
    derivatives_df['volume'] = main_df['volume'] * np.random.uniform(0.7, 1.8, size=num_rows)
else:
    derivatives_df['volume'] = np.random.uniform(1000, 10000, size=num_rows)


# Save the dummy data to a new CSV file
try:
    derivatives_df.to_csv('derivatives_data.csv', index=False)
    print("Dummy 'derivatives_data.csv' created successfully.")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")
