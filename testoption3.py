import ccxt
import pandas as pd
import numpy as np
import datetime as dt

def fetch_kraken_data_daily(days=365):
    """
    Fetch ~365 days of *daily* BTC/USD data from Kraken (via ccxt).
    Returns a DataFrame with columns: [open, high, low, close, volume], indexed by date_time.
    """
    kraken = ccxt.kraken()
    now_dt = dt.datetime.now()
    start_dt = now_dt - dt.timedelta(days=days)
    since = int(start_dt.timestamp() * 1000)
    
    # timeframe="1d" fetches daily bars
    ohlcv = kraken.fetch_ohlcv("BTC/USD", timeframe="1d", since=since)
    if not ohlcv:
        return pd.DataFrame()
    
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # Convert timestamps to Python datetimes
    df["date_time"] = pd.to_datetime(df["timestamp"], unit="ms")
    # Set the date_time column as the index
    df.set_index("date_time", inplace=True)
    # Sort by date_time ascending
    df.sort_index(inplace=True)
    
    return df

def calculate_btc_annualized_volatility_daily(df):
    """
    EXACT steps for daily realized volatility over the last 30 days:
      1) We have daily data in df (one row per day).
      2) Compute daily returns from close.
      3) Take the last 30 daily returns.
      4) Compute the sample std dev of those 30 returns.
      5) Multiply by sqrt(365) to annualize.
    """
    df_daily = df.copy()
    # 1) Ensure we have daily data (already 1d, so no resample needed, but do a safety dropna)
    df_daily.dropna(subset=["close"], inplace=True)
    if df_daily.empty:
        return np.nan
    
    # 2) Compute daily returns
    df_daily["daily_return"] = df_daily["close"].pct_change()
    
    # 3) Take the last 30 daily returns (drop the first NaN)
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if len(last_30_returns) < 1:
        return np.nan  # or do partial if you want
    
    # 4) Sample standard deviation
    daily_std = last_30_returns.std()  # sample std dev
    
    # 5) Annualize by sqrt(365)
    return daily_std * np.sqrt(365)

if __name__ == "__main__":
    # 1) Fetch daily data from Kraken
    df_kraken_daily = fetch_kraken_data_daily(days=365)
    if df_kraken_daily.empty:
        print("No daily data fetched from Kraken. Check your ccxt config or timeframe.")
    else:
        # 2) Compute the 30-day realized volatility
        rv_30 = calculate_btc_annualized_volatility_daily(df_kraken_daily)
        print(f"30-day Realized Volatility (annualized): {rv_30:.2%} (decimal {rv_30:.4f})")
