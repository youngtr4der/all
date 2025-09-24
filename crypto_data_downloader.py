import ccxt
import pandas as pd
import datetime

def download_ohlcv(symbol, timeframe, since):
    """
    Downloads historical OHLCV data for a given symbol and timeframe.

    :param symbol: The trading pair symbol (e.g., 'BTC/USDT').
    :param timeframe: The timeframe for the candles (e.g., '1h', '1d').
    :param since: The start date for the data in 'YYYY-MM-DD' format.
    :return: A pandas DataFrame with the OHLCV data, or None if an error occurs.
    """
    # Initialize the Binance.US exchange object due to location restrictions
    # NOTE: This is a workaround. The availability of pairs may differ from Binance.com.
    exchange = ccxt.binanceus({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })

    # Convert the start date string to a timestamp in milliseconds
    since_timestamp = exchange.parse8601(since + 'T00:00:00Z')

    # List to store all the fetched candles
    all_ohlcv = []

    while True:
        try:
            # Fetch OHLCV data from the exchange
            # The 'limit' parameter defines the number of candles per request (max 1000 for Binance)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=1000)

            # If no more data is returned, break the loop
            if not ohlcv:
                break

            # Add the fetched data to our list
            all_ohlcv.extend(ohlcv)

            # Update the 'since' timestamp to the timestamp of the last candle + 1 millisecond
            # This ensures the next fetch starts right after the last one, avoiding duplicates
            since_timestamp = ohlcv[-1][0] + 1

        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying...")
            continue
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    if not all_ohlcv:
        print("No data was downloaded.")
        return None

    # Convert the list of lists to a pandas DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert the timestamp column from milliseconds to a readable datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

if __name__ == '__main__':
    # --- Example Usage ---
    # Define the parameters for data download
    trading_pair = 'BTC/USDT'
    timeframe = '1h'  # 1-hour candles
    start_date = '2021-01-01'

    print(f"Downloading {timeframe} data for {trading_pair} since {start_date}...")

    # Call the download function
    ohlcv_df = download_ohlcv(trading_pair, timeframe, start_date)

    if ohlcv_df is not None and not ohlcv_df.empty:
        # Define the output CSV file name
        # e.g., BTC_USDT_1h_from_2021-01-01.csv
        filename = f"{trading_pair.replace('/', '_')}_{timeframe}_from_{start_date}.csv"

        # Save the DataFrame to a CSV file
        ohlcv_df.to_csv(filename, index=False)

        print(f"Data successfully downloaded and saved to '{filename}'")
        print("Downloaded", len(ohlcv_df), "rows of data.")
    else:
        print("Failed to download data.")
