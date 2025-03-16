import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import requests
import ccxt
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, percentileofscore
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline

###########################################
# EXPIRATION DATE SELECTION FUNCTIONS
###########################################
def get_valid_expiration_options(current_date=None):
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < 14:
        return [14, 28]
    elif current_date.day < 28:
        return [28]
    else:
        return [14, 28]

def compute_expiry_date(selected_day, current_date=None):
    if current_date is None:
        current_date = dt.datetime.now()
    try:
        if current_date.day < selected_day:
            expiry = current_date.replace(day=selected_day, hour=0, minute=0, second=0, microsecond=0)
        else:
            year = current_date.year + (current_date.month // 12)
            month = (current_date.month % 12) + 1
            expiry = dt.datetime(year, month, selected_day)
    except ValueError as e:
        st.error(f"Error computing expiry date: {e}")
        return None
    return expiry

###########################################
# Thalex API DETAILS AND GLOBAL SETTINGS
###########################################
BASE_URL = "https://thalex.com/api/v2/public"
instruments_endpoint = "instruments"
url_instruments = f"{BASE_URL}/{instruments_endpoint}"
mark_price_endpoint = "mark_price_historical_data"
url_mark_price = f"{BASE_URL}/{mark_price_endpoint}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"

def params(instrument_name):
    now = dt.datetime.now()
    start_dt = now - dt.timedelta(days=7)
    return {
        "from": int(start_dt.timestamp()),
        "to": int(now.timestamp()),
        "resolution": "5m",
        "instrument_name": instrument_name,
    }

COLUMNS = [
    "ts", "mark_price_open", "mark_price_high", "mark_price_low", "mark_price_close",
    "iv_open", "iv_high", "iv_low", "iv_close"
]

###########################################
# CREDENTIALS & LOGIN FUNCTIONS (USING TEXT FILES)
###########################################
def load_credentials():
    try:
        with open("usernames.txt", "r") as f_user:
            usernames = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            passwords = [line.strip() for line in f_pass if line.strip()]
        if len(usernames) != len(passwords):
            st.error("The number of usernames and passwords do not match.")
            return {}
        return dict(zip(usernames, passwords))
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("Please Log In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            creds = load_credentials()
            if username in creds and creds[username] == password:
                st.session_state.logged_in = True
                st.success("Logged in successfully! Click the login button a second time to open the app!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# INSTRUMENTS FETCHING & FILTERING FUNCTIONS
###########################################
def fetch_instruments():
    try:
        response = requests.get(url_instruments)
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

def get_option_instruments(instruments, option_type, expiry_str):
    filtered = [inst["instrument_name"] for inst in instruments 
                if inst["instrument_name"].startswith(f"BTC-{expiry_str}") 
                and inst["instrument_name"].endswith(f"-{option_type}")]
    return sorted(filtered)

def get_actual_iv(instrument_name):
    try:
        response = requests.get(url_mark_price, params=params(instrument_name))
        response.raise_for_status()
        data = response.json()
        marks = get_in(["result", "mark"])(data)
        if not marks:
            return None
        df = pd.DataFrame(marks, columns=COLUMNS)
        df = df.sort_values("ts")
        return df["iv_close"].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching IV for {instrument_name}: {e}")
        return None

def get_atm_iv(calls_all, spot_price):
    try:
        strike_list = [(inst, int(inst.split("-")[2])) for inst in calls_all]
        strike_list.sort(key=lambda x: abs(x[1] - spot_price))
        atm_instrument = strike_list[0][0]
        atm_iv = get_actual_iv(atm_instrument)
        return atm_iv
    except Exception as e:
        st.error(f"Error computing ATM IV: {e}")
        return None

def get_filtered_instruments(spot_price, expiry_str, t_years, multiplier=1):
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    
    atm_iv = get_atm_iv(calls_all, spot_price)
    if atm_iv is None:
        raise Exception("ATM IV could not be determined")
    
    lower_bound = spot_price * np.exp(-atm_iv * np.sqrt(t_years) * multiplier)
    upper_bound = spot_price * np.exp(atm_iv * np.sqrt(t_years) * multiplier)
    
    filtered_calls = [inst for inst in calls_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_puts = [inst for inst in puts_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    
    filtered_calls.sort(key=lambda x: int(x.split("-")[2]))
    filtered_puts.sort(key=lambda x: int(x.split("-")[2]))
    return filtered_calls, filtered_puts

###########################################
# DATA FETCHING FUNCTIONS
###########################################
@st.cache_data(ttl=30)
def fetch_data(instruments_tuple):
    instruments = list(instruments_tuple)
    try:
        df = (
            pipe(
                {name: requests.get(url_mark_price, params=params(name)) for name in instruments},
                valmap(requests.Response.json),
                valmap(get_in(["result", "mark"])),
                valmap(curry(pd.DataFrame, columns=COLUMNS)),
                valfilter(lambda df: not df.empty),
                pd.concat,
            )
            .droplevel(1)
            .reset_index(names=["instrument_name"])
            .assign(date_time=lambda df: pd.to_datetime(df["ts"], unit="s")
                    .dt.tz_localize("UTC")
                    .dt.tz_convert("America/New_York"))
            .assign(k=lambda df: df["instrument_name"].map(
                lambda s: int(s.split("-")[2]) if len(s.split("-")) >= 3 and s.split("-")[2].isdigit() else np.nan))
            .assign(option_type=lambda df: df["instrument_name"].str.split("-").str[-1])
        )
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    return df

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    try:
        params_dict = {"instrument_name": instrument_name}
        response = requests.get(URL_TICKER, params=params_dict)
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})
    except Exception as e:
        st.error(f"Error fetching ticker for {instrument_name}: {e}")
        return {}

def fetch_kraken_data():
    try:
        kraken = ccxt.kraken()
        now_dt = dt.datetime.now()
        start_dt = now_dt - dt.timedelta(days=365)
        ohlcv_5m = kraken.fetch_ohlcv("BTC/USD", timeframe="5m",
                                       since=int(start_dt.timestamp()) * 1000,
                                       limit=3000)
        ohlcv_1d = kraken.fetch_ohlcv("BTC/USD", timeframe="1d",
                                       since=int(start_dt.timestamp()) * 1000)
        df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_5m["date_time"] = pd.to_datetime(df_5m["timestamp"], unit="ms")
        df_5m = df_5m.set_index("date_time")
        df_1d = pd.DataFrame(ohlcv_1d, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_1d["date_time"] = pd.to_datetime(df_1d["timestamp"], unit="ms")
        df_1d = df_1d.set_index("date_time")
        full_df = pd.concat([df_5m, df_1d], axis=0).sort_index()
        return full_df[~full_df.index.duplicated()]
    except Exception as e:
        st.error(f"Error fetching Kraken data: {e}")
        return pd.DataFrame()

###########################################
# REALIZED VOLATILITY CALCULATION FUNCTIONS (30-day)
###########################################
def calculate_btc_annualized_volatility_daily(df):
    """
    Calculates the annualized realized volatility of BTC using the last 30 days
    of daily percentage returns.
    If a "close" column is missing, it renames "mark_price_close" to "close".
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    # Ensure a "close" column exists
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' or 'mark_price_close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    annualized_vol = daily_std * np.sqrt(365)
    return annualized_vol

def calculate_daily_realized_volatility_series(df):
    """
    Calculates a daily series of realized volatility using the daily percentage returns
    over a 30-day rolling window.
    Renames "mark_price_close" to "close" if needed.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' or 'mark_price_close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    volatility_series = df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)
    return volatility_series.dropna()

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365.25 * 86400)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T))
    return gamma

def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2)

###########################################
# REALIZED VOLATILITY - EV CALCULATION FUNCTIONS
###########################################
def compute_ev(iv, rv, T, position_side="short"):
    try:
        if position_side.lower() == "short":
            ev = (((iv**2 - rv**2) * T) / 2) * 100
        elif position_side.lower() == "long":
            ev = (((rv**2 - iv**2) * T) / 2) * 100
        else:
            ev = (((iv**2 - rv**2) * T) / 2) * 100
    except Exception as e:
        st.error(f"Error computing EV: {e}")
        ev = np.nan
    return ev

###########################################
# NORMALIZATION FOR COMPOSITE SCORE
###########################################
def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

###########################################
# SELECT OPTIMAL STRIKE - ADAPTIVE FOR SHORT vs LONG VOL
###########################################
def select_optimal_strike(ticker_list, position_side='short'):
    if not ticker_list:
        return None
    if position_side.lower() == 'short':
        weights = {"ev": 0.5, "gamma": -0.3, "oi": 0.2}
    else:
        weights = {"ev": 0.5, "gamma": 0.3, "oi": 0.2}
    ev_list = [item['EV'] for item in ticker_list]
    gamma_list = [item['gamma'] for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gamma = normalize_metrics(gamma_list)
    norm_oi = normalize_metrics(oi_list)
    best_score = -np.inf
    best_candidate = None
    for i, item in enumerate(ticker_list):
        score = (weights["ev"] * norm_ev[i] +
                 weights["gamma"] * norm_gamma[i] +
                 weights["oi"] * norm_oi[i])
        item["composite_score"] = score
        if score > best_score:
            best_score = score
            best_candidate = item
    return best_candidate

###########################################
# COMPOSITE SCORE VISUALIZATION (RAW)
###########################################
def compute_composite_score(item, position_side='short'):
    score = item['EV']
    if item.get('gamma', 0) > 0:
        if position_side.lower() == "short":
            score /= item['gamma']
        else:
            score *= item['gamma']
    score += 0.01 * item['open_interest']
    return score

###########################################
# VOLATILITY SURFACE ANALYSIS
###########################################
def plot_volatility_surface(df, spot_price):
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close',
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            spread = ask - bid
            mid = (ask + bid) / 2
            if mid > 0:
                item['EV'] *= (1 - spread / mid)
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

def recommend_volatility_strategy(atm_iv, rv):
    if atm_iv > rv:
        return "short"
    elif atm_iv < rv:
        return "long"
    else:
        return "neutral"

###########################################
# MAIN DAILY REALIZED VOLATILITY FUNCTIONS (30-day)
###########################################
def calculate_btc_annualized_volatility_daily(df):
    """
    Calculates the annualized realized volatility of BTC using the last 30 days
    of daily percentage returns.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    annualized_vol = daily_std * np.sqrt(365)
    return annualized_vol

def calculate_daily_realized_volatility_series(df):
    """
    Calculates a daily series of realized volatility using the daily percentage returns
    over a 30-day rolling window.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    volatility_series = df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)
    return volatility_series.dropna()

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365.25 * 86400)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T))
    return gamma

def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2)

###########################################
# REALIZED VOLATILITY - EV CALCULATION FUNCTIONS
###########################################
def compute_ev(iv, rv, T, position_side="short"):
    try:
        if position_side.lower() == "short":
            ev = (((iv**2 - rv**2) * T) / 2) * 100
        elif position_side.lower() == "long":
            ev = (((rv**2 - iv**2) * T) / 2) * 100
        else:
            ev = (((iv**2 - rv**2) * T) / 2) * 100
    except Exception as e:
        st.error(f"Error computing EV: {e}")
        ev = np.nan
    return ev

###########################################
# NORMALIZATION FOR COMPOSITE SCORE
###########################################
def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

###########################################
# SELECT OPTIMAL STRIKE - ADAPTIVE FOR SHORT vs LONG VOL
###########################################
def select_optimal_strike(ticker_list, position_side='short'):
    if not ticker_list:
        return None
    if position_side.lower() == 'short':
        weights = {"ev": 0.5, "gamma": -0.3, "oi": 0.2}
    else:
        weights = {"ev": 0.5, "gamma": 0.3, "oi": 0.2}
    ev_list = [item['EV'] for item in ticker_list]
    gamma_list = [item['gamma'] for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gamma = normalize_metrics(gamma_list)
    norm_oi = normalize_metrics(oi_list)
    best_score = -np.inf
    best_candidate = None
    for i, item in enumerate(ticker_list):
        score = (weights["ev"] * norm_ev[i] +
                 weights["gamma"] * norm_gamma[i] +
                 weights["oi"] * norm_oi[i])
        item["composite_score"] = score
        if score > best_score:
            best_score = score
            best_candidate = item
    return best_candidate

###########################################
# COMPOSITE SCORE VISUALIZATION (RAW)
###########################################
def compute_composite_score(item, position_side='short'):
    score = item['EV']
    if item.get('gamma', 0) > 0:
        if position_side.lower() == "short":
            score /= item['gamma']
        else:
            score *= item['gamma']
    score += 0.01 * item['open_interest']
    return score

###########################################
# VOLATILITY SURFACE ANALYSIS
###########################################
def plot_volatility_surface(df, spot_price):
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close',
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            spread = ask - bid
            mid = (ask + bid) / 2
            if mid > 0:
                item['EV'] *= (1 - spread / mid)
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

def recommend_volatility_strategy(atm_iv, rv):
    if atm_iv > rv:
        return "short"
    elif atm_iv < rv:
        return "long"
    else:
        return "neutral"

###########################################
# MAIN DAILY REALIZED VOLATILITY FUNCTIONS (30-day)
###########################################
def calculate_btc_annualized_volatility_daily(df):
    """
    Calculates the annualized realized volatility of BTC using the last 30 days
    of daily percentage returns.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    annualized_vol = daily_std * np.sqrt(365)
    return annualized_vol

def calculate_daily_realized_volatility_series(df):
    """
    Calculates a daily series of realized volatility using the daily percentage returns
    over a 30-day rolling window.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    volatility_series = df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)
    return volatility_series.dropna()

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365.25 * 86400)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T))
    return gamma

def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2)

###########################################
# REALIZED VOLATILITY - EV CALCULATION FUNCTIONS
###########################################
def compute_ev(iv, rv, T, position_side="short"):
    try:
        if position_side.lower() == "short":
            ev = (((iv**2 - rv**2) * T) / 2) * 100
        elif position_side.lower() == "long":
            ev = (((rv**2 - iv**2) * T) / 2) * 100
        else:
            ev = (((iv**2 - rv**2) * T) / 2) * 100
    except Exception as e:
        st.error(f"Error computing EV: {e}")
        ev = np.nan
    return ev

###########################################
# NORMALIZATION FOR COMPOSITE SCORE
###########################################
def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

###########################################
# SELECT OPTIMAL STRIKE - ADAPTIVE FOR SHORT vs LONG VOL
###########################################
def select_optimal_strike(ticker_list, position_side='short'):
    if not ticker_list:
        return None
    if position_side.lower() == 'short':
        weights = {"ev": 0.5, "gamma": -0.3, "oi": 0.2}
    else:
        weights = {"ev": 0.5, "gamma": 0.3, "oi": 0.2}
    ev_list = [item['EV'] for item in ticker_list]
    gamma_list = [item['gamma'] for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gamma = normalize_metrics(gamma_list)
    norm_oi = normalize_metrics(oi_list)
    best_score = -np.inf
    best_candidate = None
    for i, item in enumerate(ticker_list):
        score = (weights["ev"] * norm_ev[i] +
                 weights["gamma"] * norm_gamma[i] +
                 weights["oi"] * norm_oi[i])
        item["composite_score"] = score
        if score > best_score:
            best_score = score
            best_candidate = item
    return best_candidate

###########################################
# COMPOSITE SCORE VISUALIZATION (RAW)
###########################################
def compute_composite_score(item, position_side='short'):
    score = item['EV']
    if item.get('gamma', 0) > 0:
        if position_side.lower() == "short":
            score /= item['gamma']
        else:
            score *= item['gamma']
    score += 0.01 * item['open_interest']
    return score

###########################################
# VOLATILITY SURFACE ANALYSIS
###########################################
def plot_volatility_surface(df, spot_price):
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close',
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            spread = ask - bid
            mid = (ask + bid) / 2
            if mid > 0:
                item['EV'] *= (1 - spread / mid)
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

def recommend_volatility_strategy(atm_iv, rv):
    if atm_iv > rv:
        return "short"
    elif atm_iv < rv:
        return "long"
    else:
        return "neutral"

###########################################
# MAIN DAILY REALIZED VOLATILITY FUNCTIONS (30-day)
###########################################
def calculate_btc_annualized_volatility_daily(df):
    """
    Calculates the annualized realized volatility of BTC using the last 30 days
    of daily percentage returns.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    annualized_vol = daily_std * np.sqrt(365)
    return annualized_vol

def calculate_daily_realized_volatility_series(df):
    """
    Calculates a daily series of realized volatility using the daily percentage returns
    over a 30-day rolling window.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    volatility_series = df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)
    return volatility_series.dropna()

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365.25 * 86400)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T))
    return gamma

def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2)

###########################################
# REALIZED VOLATILITY - EV CALCULATION FUNCTIONS
###########################################
def compute_ev(iv, rv, T, position_side="short"):
    try:
        if position_side.lower() == "short":
            ev = (((iv**2 - rv**2) * T) / 2) * 100
        elif position_side.lower() == "long":
            ev = (((rv**2 - iv**2) * T) / 2) * 100
        else:
            ev = (((iv**2 - rv**2) * T) / 2) * 100
    except Exception as e:
        st.error(f"Error computing EV: {e}")
        ev = np.nan
    return ev

###########################################
# NORMALIZATION FOR COMPOSITE SCORE
###########################################
def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

###########################################
# SELECT OPTIMAL STRIKE - ADAPTIVE FOR SHORT vs LONG VOL
###########################################
def select_optimal_strike(ticker_list, position_side='short'):
    if not ticker_list:
        return None
    if position_side.lower() == 'short':
        weights = {"ev": 0.5, "gamma": -0.3, "oi": 0.2}
    else:
        weights = {"ev": 0.5, "gamma": 0.3, "oi": 0.2}
    ev_list = [item['EV'] for item in ticker_list]
    gamma_list = [item['gamma'] for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gamma = normalize_metrics(gamma_list)
    norm_oi = normalize_metrics(oi_list)
    best_score = -np.inf
    best_candidate = None
    for i, item in enumerate(ticker_list):
        score = (weights["ev"] * norm_ev[i] +
                 weights["gamma"] * norm_gamma[i] +
                 weights["oi"] * norm_oi[i])
        item["composite_score"] = score
        if score > best_score:
            best_score = score
            best_candidate = item
    return best_candidate

###########################################
# COMPOSITE SCORE VISUALIZATION (RAW)
###########################################
def compute_composite_score(item, position_side='short'):
    score = item['EV']
    if item.get('gamma', 0) > 0:
        if position_side.lower() == "short":
            score /= item['gamma']
        else:
            score *= item['gamma']
    score += 0.01 * item['open_interest']
    return score

###########################################
# VOLATILITY SURFACE ANALYSIS
###########################################
def plot_volatility_surface(df, spot_price):
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close',
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            spread = ask - bid
            mid = (ask + bid) / 2
            if mid > 0:
                item['EV'] *= (1 - spread / mid)
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

def recommend_volatility_strategy(atm_iv, rv):
    if atm_iv > rv:
        return "short"
    elif atm_iv < rv:
        return "long"
    else:
        return "neutral"

###########################################
# MAIN DAILY REALIZED VOLATILITY FUNCTIONS (30-day)
###########################################
def calculate_btc_annualized_volatility_daily(df):
    """
    Calculates the annualized realized volatility of BTC using the last 30 days
    of daily percentage returns.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    annualized_vol = daily_std * np.sqrt(365)
    return annualized_vol

def calculate_daily_realized_volatility_series(df):
    """
    Calculates a daily series of realized volatility using the daily percentage returns
    over a 30-day rolling window.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    volatility_series = df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)
    return volatility_series.dropna()

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365.25 * 86400)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T))
    return gamma

def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2)

###########################################
# REALIZED VOLATILITY - EV CALCULATION FUNCTIONS
###########################################
def compute_ev(iv, rv, T, position_side="short"):
    try:
        if position_side.lower() == "short":
            ev = (((iv**2 - rv**2) * T) / 2) * 100
        elif position_side.lower() == "long":
            ev = (((rv**2 - iv**2) * T) / 2) * 100
        else:
            ev = (((iv**2 - rv**2) * T) / 2) * 100
    except Exception as e:
        st.error(f"Error computing EV: {e}")
        ev = np.nan
    return ev

###########################################
# NORMALIZATION FOR COMPOSITE SCORE
###########################################
def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

###########################################
# SELECT OPTIMAL STRIKE - ADAPTIVE FOR SHORT vs LONG VOL
###########################################
def select_optimal_strike(ticker_list, position_side='short'):
    if not ticker_list:
        return None
    if position_side.lower() == 'short':
        weights = {"ev": 0.5, "gamma": -0.3, "oi": 0.2}
    else:
        weights = {"ev": 0.5, "gamma": 0.3, "oi": 0.2}
    ev_list = [item['EV'] for item in ticker_list]
    gamma_list = [item['gamma'] for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gamma = normalize_metrics(gamma_list)
    norm_oi = normalize_metrics(oi_list)
    best_score = -np.inf
    best_candidate = None
    for i, item in enumerate(ticker_list):
        score = (weights["ev"] * norm_ev[i] +
                 weights["gamma"] * norm_gamma[i] +
                 weights["oi"] * norm_oi[i])
        item["composite_score"] = score
        if score > best_score:
            best_score = score
            best_candidate = item
    return best_candidate

###########################################
# COMPOSITE SCORE VISUALIZATION (RAW)
###########################################
def compute_composite_score(item, position_side='short'):
    score = item['EV']
    if item.get('gamma', 0) > 0:
        if position_side.lower() == "short":
            score /= item['gamma']
        else:
            score *= item['gamma']
    score += 0.01 * item['open_interest']
    return score

###########################################
# VOLATILITY SURFACE ANALYSIS
###########################################
def plot_volatility_surface(df, spot_price):
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close',
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            spread = ask - bid
            mid = (ask + bid) / 2
            if mid > 0:
                item['EV'] *= (1 - spread / mid)
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

def recommend_volatility_strategy(atm_iv, rv):
    if atm_iv > rv:
        return "short"
    elif atm_iv < rv:
        return "long"
    else:
        return "neutral"

###########################################
# MAIN DAILY REALIZED VOLATILITY FUNCTIONS (30-day)
###########################################
def calculate_btc_annualized_volatility_daily(df):
    """
    Calculates the annualized realized volatility of BTC using the last 30 days
    of daily percentage returns.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    annualized_vol = daily_std * np.sqrt(365)
    return annualized_vol

def calculate_daily_realized_volatility_series(df):
    """
    Calculates a daily series of realized volatility using the daily percentage returns
    over a 30-day rolling window.
    """
    if "date_time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise KeyError("No 'date_time' column found and index is not a DatetimeIndex.")
    if "close" not in df.columns:
        if "mark_price_close" in df.columns:
            df = df.rename(columns={"mark_price_close": "close"})
        else:
            raise KeyError("No 'close' column found in DataFrame.")
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    volatility_series = df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)
    return volatility_series.dropna()

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365.25 * 86400)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    sigma_eff = sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T))
    return gamma

def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2)

###########################################
# REALIZED VOLATILITY - EV CALCULATION FUNCTIONS
###########################################
def compute_ev(iv, rv, T, position_side="short"):
    try:
        if position_side.lower() == "short":
            ev = (((iv**2 - rv**2) * T) / 2) * 100
        elif position_side.lower() == "long":
            ev = (((rv**2 - iv**2) * T) / 2) * 100
        else:
            ev = (((iv**2 - rv**2) * T) / 2) * 100
    except Exception as e:
        st.error(f"Error computing EV: {e}")
        ev = np.nan
    return ev

###########################################
# NORMALIZATION FOR COMPOSITE SCORE
###########################################
def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

###########################################
# SELECT OPTIMAL STRIKE - ADAPTIVE FOR SHORT vs LONG VOL
###########################################
def select_optimal_strike(ticker_list, position_side='short'):
    if not ticker_list:
        return None
    if position_side.lower() == 'short':
        weights = {"ev": 0.5, "gamma": -0.3, "oi": 0.2}
    else:
        weights = {"ev": 0.5, "gamma": 0.3, "oi": 0.2}
    ev_list = [item['EV'] for item in ticker_list]
    gamma_list = [item['gamma'] for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gamma = normalize_metrics(gamma_list)
    norm_oi = normalize_metrics(oi_list)
    best_score = -np.inf
    best_candidate = None
    for i, item in enumerate(ticker_list):
        score = (weights["ev"] * norm_ev[i] +
                 weights["gamma"] * norm_gamma[i] +
                 weights["oi"] * norm_oi[i])
        item["composite_score"] = score
        if score > best_score:
            best_score = score
            best_candidate = item
    return best_candidate

###########################################
# COMPOSITE SCORE VISUALIZATION (RAW)
###########################################
def compute_composite_score(item, position_side='short'):
    score = item['EV']
    if item.get('gamma', 0) > 0:
        if position_side.lower() == "short":
            score /= item['gamma']
        else:
            score *= item['gamma']
    score += 0.01 * item['open_interest']
    return score

###########################################
# VOLATILITY SURFACE ANALYSIS
###########################################
def plot_volatility_surface(df, spot_price):
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close',
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            spread = ask - bid
            mid = (ask + bid) / 2
            if mid > 0:
                item['EV'] *= (1 - spread / mid)
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

def recommend_volatility_strategy(atm_iv, rv):
    if atm_iv > rv:
        return "short"
    elif atm_iv < rv:
        return "long"
    else:
        return "neutral"

###########################################
# MAIN DASHBOARD FUNCTION
###########################################
def main():
    login()
    st.title("Crypto Options Dashboard - Adaptive for Short or Long Volatility")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    current_date = dt.datetime.now()
    valid_days = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_days)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Expiration date is invalid or already passed")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiry = (expiry_date - current_date).days
    T_YEARS = days_to_expiry / 365.0
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2
    
    global df_kraken
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    # Pass the original df_kraken to the volatility function (it will be resampled internally)
    rv = calculate_btc_annualized_volatility_daily(df_kraken)
    st.write(f"Computed Realized Volatility (annualized, 30-day): {rv:.4f}")
    
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    atm_iv = get_atm_iv(calls_all, spot_price)
    if atm_iv is None:
        st.error("Could not compute ATM IV.")
        return
    st.write(f"ATM Implied Volatility: {atm_iv:.4f}")
    
    recommended_strategy = recommend_volatility_strategy(atm_iv, rv)
    st.write(f"### Automatically Recommended Volatility Strategy: {recommended_strategy.upper()} VOL")
    if recommended_strategy == "short":
        st.write("Reason: ATM IV is higher than realized volatility (RV).")
    elif recommended_strategy == "long":
        st.write("Reason: ATM IV is lower than realized volatility (RV).")
    else:
        st.write("Reason: ATM IV and RV are approximately equal.")
    
    position_side = st.sidebar.selectbox("Volatility Strategy", ["short", "long"],
                                           index=0 if recommended_strategy=="short" else 1)
    st.sidebar.write(f"Selected strategy: {position_side}")
    
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts
    
    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return
    
    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")
    
    df_iv_agg = (df.groupby("date_time", as_index=False)["iv_close"]
                 .mean().rename(columns={"iv_close": "iv_mean"}))
    df_iv_agg["date_time"] = pd.to_datetime(df_iv_agg["date_time"])
    df_iv_agg = df_iv_agg.set_index("date_time")
    df_iv_agg = df_iv_agg.resample("5min").mean().ffill().sort_index()
    df_iv_agg["iv_rolling_mean"] = df_iv_agg["iv_mean"].rolling("1D").mean()
    df_iv_agg["iv_rolling_std"] = df_iv_agg["iv_mean"].rolling("1D").std()
    df_iv_agg["upper_zone"] = df_iv_agg["iv_rolling_mean"] + df_iv_agg["iv_rolling_std"]
    df_iv_agg["lower_zone"] = df_iv_agg["iv_rolling_mean"] - df_iv_agg["iv_rolling_std"]
    df_iv_agg_reset = df_iv_agg.reset_index()
    
    preliminary_ticker_list = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if not ticker_data or "open_interest" not in ticker_data:
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        raw_iv = ticker_data.get("iv", None)
        if raw_iv is None:
            continue
        preliminary_ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "iv": raw_iv
        })
    # Inline definitions for smile and ticker list builders:
    def build_smile_df(ticker_list):
        df = pd.DataFrame(ticker_list)
        df = df.dropna(subset=["iv"])
        smile_df = df.groupby("strike", as_index=False)["iv"].mean()
        return smile_df
    def build_ticker_list(all_instruments, spot, T, smile_df):
        ticker_list = []
        for instrument in all_instruments:
            ticker_data = fetch_ticker(instrument)
            if not (ticker_data and "open_interest" in ticker_data):
                continue
            try:
                strike = int(instrument.split("-")[2])
            except Exception:
                continue
            option_type = instrument.split("-")[-1]
            raw_iv = ticker_data.get("iv", None)
            if raw_iv is None:
                continue
            adjusted_iv = np.interp(strike, smile_df["strike"].values, smile_df["iv"].values)
            try:
                d1 = (np.log(spot / strike) + 0.5 * adjusted_iv**2 * T) / (adjusted_iv * np.sqrt(T))
            except Exception:
                continue
            delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
            ticker_list.append({
                "instrument": instrument,
                "strike": strike,
                "option_type": option_type,
                "open_interest": ticker_data["open_interest"],
                "delta": delta_est,
                "iv": adjusted_iv
            })
        return ticker_list
    smile_df = build_smile_df(preliminary_ticker_list)
    global ticker_list
    ticker_list = build_ticker_list(all_instruments, spot_price, T_YEARS, smile_df)
    
    daily_rv_series = calculate_daily_realized_volatility_series(df_kraken)
    daily_rv = daily_rv_series.tolist()
    daily_iv = df_iv_agg["iv_mean"].resample("D").mean().dropna().tolist()
    def compute_historical_vrp(daily_iv, daily_rv):
        n = min(len(daily_iv), len(daily_rv))
        return [(iv**2) - (rv**2) for iv, rv in zip(daily_iv[:n], daily_rv[:n])]
    historical_vrps = compute_historical_vrp(daily_iv, daily_rv)
    
    st.subheader("Volatility Trading Decision Tool")
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance",
                                          options=["Conservative", "Moderate", "Aggressive"],
                                          index=1)
    
    def evaluate_trade_strategy(df, spot_price, risk_tolerance, df_iv_agg_reset,
                                historical_vols, historical_vrps, expiry_date):
        iv_vol = df["iv_close"].mean() if not df.empty else np.nan
        rv_vol = calculate_btc_annualized_volatility_daily(df)
        vol_regime = "Risk-On" if iv_vol < rv_vol else "Risk-Off"
        vrp_regime = "Long Volatility" if (iv_vol**2 - rv_vol**2) < 0 else "Short Volatility"
        put_oi = df[df["option_type"]=="P"]["open_interest"].sum() if not df.empty else 0
        call_oi = df[df["option_type"]=="C"]["open_interest"].sum() if not df.empty else 0
        put_call_ratio = put_oi / call_oi if call_oi > 0 else np.inf
        avg_call_delta = 0.0
        avg_put_delta = 0.0
        avg_call_gamma = 0.0
        avg_put_gamma = 0.0
        return {
            "iv": iv_vol,
            "rv": rv_vol,
            "vol_regime": vol_regime,
            "vrp_regime": vrp_regime,
            "put_call_ratio": put_call_ratio,
            "avg_call_delta": avg_call_delta,
            "avg_put_delta": avg_put_delta,
            "avg_call_gamma": avg_call_gamma,
            "avg_put_gamma": avg_put_gamma,
            "recommendation": "Demo Recommendation",
            "position": "Demo Position",
            "hedge_action": "Demo Hedge Action"
        }
    
    trade_decision = evaluate_trade_strategy(
        df, spot_price, risk_tolerance, df_iv_agg_reset,
        historical_vols=daily_rv_series,
        historical_vrps=historical_vrps,
        expiry_date=expiry_date
    )
    
    st.write("### Market and Volatility Metrics")
    st.write(f"Implied Volatility (IV): {trade_decision['iv']:.2%}")
    st.write(f"Realized Volatility (RV): {trade_decision['rv']:.2%}")
    st.write(f"Market Regime: {trade_decision['vol_regime']}")
    st.write(f"VRP Regime: {trade_decision['vrp_regime']}")
    st.write(f"Put/Call Open Interest Ratio: {trade_decision['put_call_ratio']:.2f}")
    st.write(f"Average Call Delta: {trade_decision['avg_call_delta']:.4f}")
    st.write(f"Average Put Delta: {trade_decision['avg_put_delta']:.4f}")
    st.write(f"Average Gamma: {trade_decision['avg_call_gamma']:.6f}")
    
    st.subheader("Trading Recommendation")
    st.write(f"**Recommendation:** {trade_decision['recommendation']}")
    st.write(f"**Position:** {trade_decision['position']}")
    st.write(f"**Hedge Action:** {trade_decision['hedge_action']}")
    
    rv_series = calculate_daily_realized_volatility_series(df_kraken)
    rv_scalar = rv_series.iloc[-1] if not rv_series.empty else np.nan
    position = trade_decision['position']
    st.write("EV analysis not fully implemented in this demo.")
    
    if st.button("Simulate Trade"):
        st.write("Simulating trade based on recommendation...")
        st.write("Position Size: Adjust based on capital (e.g., 1-5% of portfolio for chosen risk tolerance)")
        st.write("Monitor price and volatility in real-time and adjust hedges dynamically.")
    
    st.subheader("Volatility Smile at Latest Timestamp")
    latest_ts = df["date_time"].max()
    smile_df_latest = df[df["date_time"] == latest_ts]
    if not smile_df_latest.empty:
        atm_strike = smile_df_latest.loc[smile_df_latest["mark_price_close"].idxmax(), "k"]
        smile_df_latest = smile_df_latest.sort_values(by="k")
        fig_vol_smile = px.line(smile_df_latest, x="k", y="iv_close", markers=True,
                                title=f"Volatility Smile at {latest_ts.strftime('%d %b %H:%M')}",
                                labels={"iv_close": "IV", "k": "Strike"})
        cheap_hedge_strike = smile_df_latest.loc[smile_df_latest["iv_close"].idxmin(), "k"]
        fig_vol_smile.add_vline(x=cheap_hedge_strike, line=dict(dash="dash", color="green"),
                                annotation_text=f"Cheap Hedge ({cheap_hedge_strike})", annotation_position="top")
        fig_vol_smile.add_vline(x=spot_price, line=dict(dash="dash", color="blue"),
                                annotation_text=f"Price: {spot_price:.2f}", annotation_position="bottom left")
        fig_vol_smile.update_layout(height=400, width=600)
        st.plotly_chart(fig_vol_smile, use_container_width=True)
        def plot_gamma_heatmap(df):
            st.subheader("Gamma Heatmap by Strike and Time")
            fig = px.density_heatmap(df, x="date_time", y="k", z="gamma",
                                     color_continuous_scale="Viridis", title="Gamma by Strike Over Time")
            fig.update_layout(height=400, width=800)
            st.plotly_chart(fig, use_container_width=True)
        plot_gamma_heatmap(pd.concat([df_calls, df_puts]))
    
    gex_data = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if ticker_data and "open_interest" in ticker_data:
            oi = ticker_data["open_interest"]
        else:
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        if option_type == "C":
            candidate = df_calls[df_calls["instrument_name"] == instrument]
        else:
            candidate = df_puts[df_puts["instrument_name"] == instrument]
        if candidate.empty:
            continue
        row = candidate.iloc[0]
        gex = compute_gex(row, spot_price, oi)
        gex_data.append({"strike": strike, "gex": gex, "option_type": option_type})
    df_gex = pd.DataFrame(gex_data)
    if not df_gex.empty:
        def plot_gex_by_strike(df):
            st.subheader("Gamma Exposure (GEX) by Strike")
            fig = px.bar(df, x="strike", y="gex", color="option_type",
                         title="Gamma Exposure (GEX) by Strike", labels={"gex": "GEX", "strike": "Strike Price"})
            fig.update_layout(height=400, width=800)
            st.plotly_chart(fig, use_container_width=True)
        def plot_net_gex(df, spot_price):
            st.subheader("Net Gamma Exposure by Strike")
            df_net = df.groupby("strike").apply(
                lambda x: x.loc[x["option_type"]=="C", "gex"].sum() - x.loc[x["option_type"]=="P", "gex"].sum()
            ).reset_index(name="net_gex")
            df_net["sign"] = df_net["net_gex"].apply(lambda val: "Negative" if val < 0 else "Positive")
            fig = px.bar(df_net, x="strike", y="net_gex", color="sign",
                         title="Net Gamma Exposure (Calls GEX - Puts GEX)",
                         labels={"net_gex": "Net GEX", "strike": "Strike Price"})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.add_vline(x=spot_price, line_dash="dash", line_color="lightgrey",
                          annotation_text=f"Spot {spot_price:.0f}", annotation_position="top right")
            fig.update_layout(height=400, width=800)
            st.plotly_chart(fig, use_container_width=True)
        plot_gex_by_strike(df_gex)
        plot_net_gex(df_gex, spot_price)

if __name__ == '__main__':
    main()
