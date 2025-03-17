import streamlit as st
import datetime as dt
import pandas as pd
import requests
import numpy as np
import ccxt
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import math

###########################################
# REALIZED VOLATILITY FUNCTIONS
###########################################
def calculate_ewma_roger_satchell_volatility(price_data, span):
    """
    EWMA Roger–Satchell Volatility:
    
    Step 1: Compute 'rs' for each observation:
           rs = log(high/close) * log(high/open) + log(low/close) * log(low/open)
    Step 2: Apply an EWMA with the given span (e.g., days to expiry).
    Step 3: Clip negative EWMA values to zero.
    Step 4: Take the square root to yield the realized volatility.
    """
    df = price_data.copy()
    df['rs'] = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
    ewma_rs = df['rs'].ewm(span=span, adjust=False).mean()
    return np.sqrt(ewma_rs.clip(lower=0))

def compute_realized_volatility_5min(df, annualize_days=365):
    """
    5-Minute Interval Realized Volatility:
    
    Step 1: Compute 'rs' for each 5-minute interval.
    Step 2: Sum the rs values to obtain total variance.
    Step 3: Calculate an annualization factor based on 12 intervals per hour over 24 hours.
    Step 4: Return the square root of total variance multiplied by the annualization factor.
    """
    df = df.copy()
    df['rs'] = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
    total_variance = df['rs'].sum()
    if total_variance <= 0:
        return 0.0
    N = len(df)
    M = annualize_days * 24 * 12  # 12 intervals per hour * 24 hours
    annualization_factor = np.sqrt(M / N)
    return np.sqrt(total_variance) * annualization_factor

def calculate_btc_annualized_volatility_daily(df):
    """
    Daily Annualized Volatility for BTC:
    
    - Resample BTC price data to daily frequency.
    - Compute daily returns.
    - Use the last 30 days' returns to compute the sample standard deviation.
    - Annualize by multiplying by sqrt(365).
    
    Returns a single volatility value based on the last 30 days.
    """
    if "date_time" not in df.columns:
        df = df.reset_index()
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    return daily_std * np.sqrt(365)

def calculate_daily_realized_volatility_series_5d(df):
    """
    Daily Realized Volatility Series (5-day rolling):
    
    - Resample BTC price data to daily frequency.
    - Compute daily returns.
    - Return a 5-day rolling standard deviation (annualized by sqrt(365)) of daily returns.
    
    Returns a Series indexed by date.
    """
    if "date_time" not in df.columns:
        df = df.reset_index()
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    return df_daily["daily_return"].rolling(window=5).std() * np.sqrt(365)

###########################################
# EXPIRATION DATE SELECTION FUNCTIONS
###########################################
def get_valid_expiration_options(current_date=None):
    """Return valid expiration day options based on today's date."""
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < 14:
        return [14, 28]
    elif current_date.day < 28:
        return [28]
    else:
        return [14, 28]

def compute_expiry_date(selected_day, current_date=None):
    """
    Compute the expiration date based on the selected day.
    If today is before the selected day, use the current month;
    otherwise, roll over to the next month.
    """
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < selected_day:
        try:
            expiry = current_date.replace(day=selected_day, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            st.error("Invalid expiration date for current month.")
            return None
    else:
        year = current_date.year + (current_date.month // 12)
        month = (current_date.month % 12) + 1
        try:
            expiry = dt.datetime(year, month, selected_day)
        except ValueError:
            st.error("Invalid expiration date for next month.")
            return None
    return expiry

###########################################
# THALEX API DETAILS & CONFIGURATION
###########################################
BASE_URL = "https://thalex.com/api/v2/public"
instruments_endpoint = "instruments"  # For fetching available instruments
url_instruments = f"{BASE_URL}/{instruments_endpoint}"
mark_price_endpoint = "mark_price_historical_data"
url_mark_price = f"{BASE_URL}/{mark_price_endpoint}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"

# Example rolling window configuration (e.g., 7-day window)
windows = {"7D": "vrp_7d"}

def params(instrument_name):
    now = dt.datetime.now()
    start_dt = now - dt.timedelta(days=7)
    return {
        "from": int(start_dt.timestamp()),
        "to": int(now.timestamp()),
        "resolution": "5m",
        "instrument_name": instrument_name,
    }

# Expected column names from Thalex API mark price data
COLUMNS = [
    "ts",
    "mark_price_open",
    "mark_price_high",
    "mark_price_low",
    "mark_price_close",
    "iv_open",
    "iv_high",
    "iv_low",
    "iv_close",
]

###########################################
# CREDENTIALS & LOGIN FUNCTIONS
###########################################
def load_credentials():
    """
    Load user credentials from text files.
    Returns a dictionary mapping username to password.
    """
    try:
        with open("usernames.txt", "r") as f_user:
            usernames = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            passwords = [line.strip() for line in f_pass if line.strip()]
        if len(usernames) != len(passwords):
            st.error("The number of usernames and passwords do not match.")
            return {}
        creds = dict(zip(usernames, passwords))
        return creds
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    """
    Display a login form and validate credentials.
    The dashboard loads only after successful authentication.
    """
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
                st.success("Logged in successfully! Click Login again to connect to the option app.")
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# INSTRUMENTS FETCHING & FILTERING FUNCTIONS
###########################################
def fetch_instruments():
    """Fetch the instruments list from the Thalex API."""
    response = requests.get(url_instruments)
    if response.status_code != 200:
        raise Exception("Failed to fetch instruments")
    data = response.json()
    return data.get("result", [])

def get_option_instruments(instruments, option_type, expiry_str):
    """
    Filter instruments for options (calls or puts) for BTC with the specified expiry.
    Option type should be 'C' for calls or 'P' for puts.
    """
    filtered = [
        inst["instrument_name"] for inst in instruments
        if inst["instrument_name"].startswith(f"BTC-{expiry_str}") and inst["instrument_name"].endswith(f"-{option_type}")
    ]
    return sorted(filtered)

def get_actual_iv(instrument_name):
    """
    Fetch mark price data for the given instrument and return its latest iv_close value.
    """
    response = requests.get(url_mark_price, params=params(instrument_name))
    if response.status_code != 200:
        return None
    data = response.json()
    marks = get_in(["result", "mark"])(data)
    if not marks:
        return None
    df = pd.DataFrame(marks, columns=COLUMNS)
    df = df.sort_values("ts")
    return df["iv_close"].iloc[-1]

def get_filtered_instruments(spot_price, expiry_str, t_years, multiplier=1):
    """
    Filter instruments based on a theoretical strike range derived from a standard deviation move.
    """
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    
    strike_list = [(inst, int(inst.split("-")[2])) for inst in calls_all]
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
    closest_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    nearest_instrument = strike_list[closest_index][0]
    
    actual_iv = get_actual_iv(nearest_instrument)
    if actual_iv is None:
        raise Exception("Could not fetch actual IV for the nearest instrument")
    
    lower_bound = spot_price * np.exp(-actual_iv * np.sqrt(t_years) * multiplier)
    upper_bound = spot_price * np.exp(actual_iv * np.sqrt(t_years) * multiplier)
    
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
    """
    Fetch Thalex mark price data for the provided instruments over the past 7 days at 5m resolution.
    """
    instruments = list(instruments_tuple)
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
        .assign(k=lambda df: df["instrument_name"].map(lambda s: int(s.split("-")[2])
                                                      if len(s.split("-")) >= 3 and s.split("-")[2].isdigit() else np.nan))
        .assign(option_type=lambda df: df["instrument_name"].str.split("-").str[-1])
    )
    return df

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    """
    Fetch ticker data for the given instrument.
    """
    params_dict = {"instrument_name": instrument_name}
    response = requests.get(URL_TICKER, params=params_dict)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("result", {})

def fetch_kraken_data():
    """
    Fetch 7 days of 5m BTC/USD data from Kraken (via ccxt).
    """
    kraken = ccxt.kraken()
    now_dt = dt.datetime.now()
    start_dt = now_dt - dt.timedelta(days=7)
    since = int(start_dt.timestamp() * 1000)
    ohlcv = kraken.fetch_ohlcv("BTC/USD", timeframe="5m", since=since, limit=3000)
    df_kraken = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df_kraken.empty:
        return pd.DataFrame()
    df_kraken["date_time"] = pd.to_datetime(df_kraken["timestamp"], unit="ms")
    df_kraken["date_time"] = df_kraken["date_time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df_kraken = df_kraken.sort_values(by="date_time").reset_index(drop=True)
    cutoff_start = (now_dt - dt.timedelta(days=7)).astimezone(df_kraken["date_time"].dt.tz)
    df_kraken = df_kraken[df_kraken["date_time"] >= cutoff_start]
    return df_kraken

###########################################
# DUMMY RISK ADJUSTMENT FUNCTION
###########################################
def compute_risk_adjustment_factor_cf(df, alpha=0):
    """
    Dummy implementation: compute risk adjustment factor as BTC's annualized volatility
    scaled by (1 + alpha). Adjust this function as needed.
    """
    btc_vol = calculate_btc_annualized_volatility_daily(df)
    return btc_vol * (1 + alpha)

###########################################
# ROLLING VRP FUNCTION
###########################################
def compute_rolling_vrp(group, window_str):
    """
    Compute rolling variance risk premium (VRP) for a given group over the specified window.
    VRP = (rolling average of squared IV) - (rolling sum of squared log returns)
    """
    rolling_rv = group["log_return"].expanding(min_periods=1).apply(lambda x: np.nansum(x**2), raw=True)
    rolling_iv = group["iv_close"].rolling(window_str, min_periods=1).apply(lambda x: np.mean(x**2), raw=True)
    return rolling_iv - rolling_rv

###########################################
# OPTION DELTA CALCULATION FUNCTION
###########################################
def compute_delta(row, S):
    """
    Compute the Black-Scholes delta for an option.
    Uses the instrument's expiration date parsed from its name to calculate time to expiry.
    """
    try:
        expiry_str = row['instrument_name'].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        # Assume expiration is given in UTC; adjust if necessary.
        expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return np.nan
    now = dt.datetime.now(tz=dt.timezone.utc)
    T = (expiry_date - now).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        T = 0.0001
    K = row['k']
    sigma = row['iv_close']
    if sigma <= 0:
        return np.nan
    try:
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row['option_type'] == 'C' else norm.cdf(d1) - 1

###########################################
# DELTA-BASED DYNAMIC REGIME FUNCTIONS
###########################################
def compute_average_delta(df_calls, df_puts, S):
    """
    Compute average call and put delta for each timestamp.
    """
    if "delta" not in df_calls.columns:
        df_calls["delta"] = df_calls.apply(lambda row: compute_delta(row, S), axis=1)
    if "delta" not in df_puts.columns:
        df_puts["delta"] = df_puts.apply(lambda row: compute_delta(row, S), axis=1)
    
    df_calls_mean = (
        df_calls.groupby("date_time", as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "call_delta_avg"})
    )
    df_puts_mean = (
        df_puts.groupby("date_time", as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "put_delta_avg"})
    )
    df_merged = pd.merge(df_calls_mean, df_puts_mean, on="date_time", how="outer").sort_values("date_time")
    df_merged["delta_diff"] = df_merged["call_delta_avg"] - df_merged["put_delta_avg"]
    return df_merged

def rolling_percentile_zones(df, column="delta_diff", window="1D", lower_percentile=30, upper_percentile=70):
    """
    Compute rolling percentile zones for the specified column.
    """
    df = df.set_index("date_time").sort_index()
    def percentile_in_window(x, q):
        return np.percentile(x, q)
    df["rolling_lower_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, lower_percentile), raw=False)
    df["rolling_upper_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, upper_percentile), raw=False)
    return df.reset_index()

def classify_regime(row):
    """
    Classify regime based on delta_diff relative to rolling zones.
    """
    if pd.isna(row["rolling_lower_zone"]) or pd.isna(row["rolling_upper_zone"]):
        return "Neutral"
    if row["delta_diff"] > row["rolling_upper_zone"]:
        return "Risk-On"
    elif row["delta_diff"] < row["rolling_lower_zone"]:
        return "Risk-Off"
    else:
        return "Neutral"

###########################################
# GAMMA & GAMMA EXPOSURE FUNCTIONS
###########################################
def compute_gamma(row, S):
    """
    Compute the Black-Scholes gamma for an option.
    """
    try:
        expiry_str = row['instrument_name'].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return np.nan
    now = dt.datetime.now(tz=dt.timezone.utc)
    T = (expiry_date - now).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row['k']
    sigma = row['iv_close']
    if sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def compute_gex(row, S, oi):
    """
    Compute Gamma Exposure (GEX) for an option.
    """
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2) * 0.01

def plot_gamma_heatmap(df):
    st.subheader("Gamma Heatmap by Strike and Time")
    fig_gamma_heatmap = px.density_heatmap(
        df,
        x="date_time",
        y="k",
        z="gamma",
        color_continuous_scale="Viridis",
        title="Gamma by Strike Over Time"
    )
    fig_gamma_heatmap.update_layout(height=400, width=800)
    st.plotly_chart(fig_gamma_heatmap, use_container_width=True)

def plot_gex_by_strike(df_gex):
    st.subheader("Gamma Exposure (GEX) by Strike")
    fig_gex = px.bar(
        df_gex,
        x="strike",
        y="gex",
        color="option_type",
        title="Gamma Exposure (GEX) by Strike",
        labels={"gex": "GEX", "strike": "Strike Price"}
    )
    fig_gex.update_layout(height=400, width=800)
    st.plotly_chart(fig_gex, use_container_width=True)

def plot_net_gex(df_gex, spot_price):
    st.subheader("Net Gamma Exposure by Strike")
    df_gex_net = (
        df_gex.groupby("strike", group_keys=False)
              .apply(lambda x: x.loc[x["option_type"] == "C", "gex"].sum() - x.loc[x["option_type"] == "P", "gex"].sum())
              .reset_index(name="net_gex")
    )
    df_gex_net["sign"] = df_gex_net["net_gex"].apply(lambda val: "Negative" if val < 0 else "Positive")
    fig_net_gex = px.bar(
        df_gex_net,
        x="strike",
        y="net_gex",
        color="sign",
        color_discrete_map={"Negative": "orange", "Positive": "blue"},
        title="Net Gamma Exposure (Calls GEX - Puts GEX)",
        labels={"net_gex": "Net GEX", "strike": "Strike Price"}
    )
    fig_net_gex.add_hline(y=0, line_dash="dash", line_color="red")
    fig_net_gex.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="lightgrey",
        annotation_text=f"Spot {spot_price:.0f}",
        annotation_position="top right"
    )
    fig_net_gex.update_layout(height=400, width=800)
    st.plotly_chart(fig_net_gex, use_container_width=False)

###########################################
# GVOL_DIRECTION & TICKER FUNCTIONS
###########################################
def compute_gvol_direction(row):
    """
    Compute the trade direction (GVOL_DIRECTION) using a heuristic algorithm.
    """
    score = 0
    if row["delta"] > 0:
        score += 0.5
    else:
        score -= 0.5
    if row.get("open_interest", 0) > 100:
        score += 0.2
    else:
        score -= 0.2
    score += np.random.uniform(-0.05, 0.05)
    score *= 10  # scale factor
    return 1 if score >= 0 else -1

def build_ticker_list(df, df_calls, df_puts, spot_price):
    """
    Build a ticker list from the Thalex data.
    Instead of a fixed T_est, calculate time to expiry from the instrument's expiration date.
    """
    ticker_list = []
    for instrument in df["instrument_name"].unique():
        if not isinstance(instrument, str):
            continue
        try:
            parts = instrument.split("-")
            strike = int(parts[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        ticker_data = fetch_ticker(instrument)
        if not ticker_data or "open_interest" not in ticker_data:
            continue
        oi = ticker_data["open_interest"]
        iv = ticker_data.get("iv", None)
        if iv is None:
            continue
        try:
            expiry_str = instrument.split("-")[1]
            expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
            expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
            now = dt.datetime.now(tz=dt.timezone.utc)
            T_est = (expiry_date - now).total_seconds() / (365 * 24 * 3600)
            if T_est <= 0:
                T_est = 0.0001
        except Exception:
            T_est = 0.05
        S = spot_price
        try:
            d1 = (np.log(S / strike) + 0.5 * iv**2 * T_est) / (iv * np.sqrt(T_est))
        except Exception:
            continue
        delta_val = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        gamma_val = np.nan
        if option_type == "C":
            temp = df_calls[df_calls["instrument_name"] == instrument]
            if not temp.empty:
                gamma_val = compute_gamma(temp.iloc[0], spot_price)
        else:
            temp = df_puts[df_puts["instrument_name"] == instrument]
            if not temp.empty:
                gamma_val = compute_gamma(temp.iloc[0], spot_price)
        if np.isnan(gamma_val):
            continue
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": oi,
            "delta": delta_val,
            "gamma": gamma_val
        })
    return ticker_list

###########################################
# MAIN DASHBOARD FUNCTION
###########################################
def main():
    # Ensure user is logged in before proceeding
    login()
    
    st.title("Crypto Options Visualization Dashboard (Plotly Version)")
    
    # Optional Logout Button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    # -------------------------------
    # EXPIRATION DATE SELECTION
    # -------------------------------
    current_date = dt.datetime.now()
    valid_options = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_options)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Wrong expiration date already passed")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiry = (expiry_date - current_date).days
    T_YEARS = days_to_expiry / 365  # Convert days to years
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    # -------------------------------
    # DEVIATION RANGE SELECTION
    # -------------------------------
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2

    # -------------------------------
    # FETCH MARKET DATA
    # -------------------------------
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts
    
    # First attempt to fetch data from Thalex
    df = fetch_data(tuple(all_instruments))
    
    # Fallback: If no data and selected day is 14, switch to 28
    if df.empty and selected_day == 14:
        st.warning("No data found for the 14th. Switching to the 28th.")
        selected_day = 28
        expiry_date = compute_expiry_date(selected_day, current_date)
        expiry_str = expiry_date.strftime("%d%b%y").upper()
        st.sidebar.markdown(f"**Using Expiration Date (fallback):** {expiry_str}")
        try:
            filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
        except Exception as e:
            st.error(f"Error fetching instruments with fallback: {e}")
            return
        all_instruments = filtered_calls + filtered_puts
        df = fetch_data(tuple(all_instruments))
    
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return
    
    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")
    
    df_kraken_renamed = df_kraken.rename(columns={
        "open": "spot_open", "high": "spot_high", "low": "spot_low", "close": "spot_close"
    })
    
    first_instrument = all_instruments[0]
    df_mark = df[df["instrument_name"] == first_instrument].copy().sort_values("date_time")
    df_merged = pd.merge(
        df_mark,
        df_kraken_renamed[["date_time", "spot_open", "spot_high", "spot_low", "spot_close"]],
        on="date_time", how="inner"
    ).sort_values("date_time")
    
    # -------------------------------
    # CHARTS: CANDLESTICK & MARK PRICE
    # -------------------------------
    st.subheader("Candlestick Chart (Single Instrument - Mark Price)")
    df_candle = df_mark.sort_values(by="date_time")
    fig_candlestick = go.Figure(data=[go.Candlestick(
        x=df_candle["date_time"],
        open=df_candle["mark_price_open"],
        high=df_candle["mark_price_high"],
        low=df_candle["mark_price_low"],
        close=df_candle["mark_price_close"]
    )])
    fig_candlestick.update_layout(
        title="Candlestick Chart (Single Instrument)",
        xaxis_title="Date",
        yaxis=dict(showticklabels=False, title=""),
        width=1000,
        height=600
    )
    st.plotly_chart(fig_candlestick)
    
    st.subheader("Mark Price with Mean Line - Calls")
    if not df_calls.empty:
        date_range_str_calls = f"{df_calls['date_time'].iloc[0].strftime('%d %b %y %I:%M %p')} — {df_calls['date_time'].iloc[-1].strftime('%d %b %y %I:%M %p')}"
        mean_mark_calls = df_calls['mark_price_close'].mean()
        fig_mark_price_calls = go.Figure()
        for instrument in df_calls['instrument_name'].unique():
            df_inst = df_calls[df_calls['instrument_name'] == instrument]
            fig_mark_price_calls.add_trace(
                go.Scatter(
                    x=df_inst['date_time'],
                    y=df_inst['mark_price_close'],
                    mode='lines',
                    line=dict(width=2),
                    opacity=0.8,
                    name=instrument
                )
            )
        fig_mark_price_calls.add_shape(
            type="line",
            x0=df_calls['date_time'].min(),
            x1=df_calls['date_time'].max(),
            y0=mean_mark_calls,
            y1=mean_mark_calls,
            line=dict(dash="dash", color="firebrick"),
            xref="x",
            yref="y"
        )
        fig_mark_price_calls.update_layout(
            height=400,
            width=800,
            xaxis_title="Date",
            yaxis_title="Mark Price (Calls)",
            showlegend=True,
            title_text=f"Mark Price with Mean Line - Calls\n{date_range_str_calls}"
        )
        fig_mark_price_calls.update_xaxes(tickformat="%m/%d %H:%M")
        st.plotly_chart(fig_mark_price_calls, use_container_width=True)
    else:
        st.warning("No call data available.")
    
    st.subheader("Mark Price with Mean Line - Puts")
    if not df_puts.empty:
        date_range_str_puts = f"{df_puts['date_time'].iloc[0].strftime('%d %b %y %I:%M %p')} — {df_puts['date_time'].iloc[-1].strftime('%d %b %y %I:%M %p')}"
        mean_mark_puts = df_puts['mark_price_close'].mean()
        fig_mark_price_puts = go.Figure()
        for instrument in df_puts['instrument_name'].unique():
            df_inst = df_puts[df_puts['instrument_name'] == instrument]
            fig_mark_price_puts.add_trace(
                go.Scatter(
                    x=df_inst['date_time'],
                    y=df_inst['mark_price_close'],
                    mode='lines',
                    line=dict(width=2),
                    opacity=0.8,
                    name=instrument
                )
            )
        fig_mark_price_puts.add_shape(
            type="line",
            x0=df_puts['date_time'].min(),
            x1=df_puts['date_time'].max(),
            y0=mean_mark_puts,
            y1=mean_mark_puts,
            line=dict(dash="dash", color="firebrick"),
            xref="x",
            yref="y"
        )
        fig_mark_price_puts.update_layout(
            height=400,
            width=800,
            xaxis_title="Date",
            yaxis_title="Mark Price (Puts)",
            showlegend=True,
            title_text=f"Mark Price with Mean Line - Puts\n{date_range_str_puts}"
        )
        fig_mark_price_puts.update_xaxes(tickformat="%m/%d %H:%M")
        st.plotly_chart(fig_mark_price_puts, use_container_width=True)
    else:
        st.warning("No put data available.")    
    
    st.subheader("IV with Optimal Hedge Zone")
    fig_iv_mean = px.line(
        df,
        x='date_time',
        y='iv_close',
        color='instrument_name',
        title="IV with Optimal Hedge Zone"
    )
    fig_iv_mean.update_traces(line=dict(width=2), opacity=0.8)
    mean_iv = df['iv_close'].mean()
    fig_iv_mean.add_hline(
        y=mean_iv,
        line_dash='dash',
        line_color='firebrick',
        annotation_text='Mean IV',
        annotation_position='top left'
    )
    fig_iv_mean.update_layout(
        width=800,
        height=400,
        xaxis_title="Date",
        yaxis_title="IV"
    )
    fig_iv_mean.update_xaxes(tickformat="%m/%d %H:%M")
    st.plotly_chart(fig_iv_mean, use_container_width=True)
    
    # -------------------------------------------------------------------
    # IV vs RV Comparison (Daily) -- Historical RV Series (5-day rolling)
    # -------------------------------------------------------------------
    st.subheader("Comparison: Implied Volatility vs. Realized Volatility (Daily)")
    # Resample options IV data to daily average using the "iv_close" column
    df_iv_daily = df[["date_time", "iv_close"]].copy().set_index("date_time").resample("D").mean().reset_index()
    df_iv_daily = df_iv_daily.rename(columns={"iv_close": "iv_mean"})
    
    # Compute the daily historical RV series using a 5-day rolling window from Kraken data
    df_rv_daily = calculate_daily_realized_volatility_series_5d(df_kraken).reset_index()
    df_rv_daily = df_rv_daily.rename(columns={"daily_return": "rv"})
    
    # Merge the IV and RV data on date using an asof merge
    df_iv_rv = pd.merge_asof(df_iv_daily.sort_values("date_time"),
                              df_rv_daily.sort_values("date_time"),
                              on="date_time", direction="backward")
    
    fig_iv_rv = go.Figure()
    fig_iv_rv.add_trace(go.Scatter(
        x=df_iv_rv["date_time"],
        y=df_iv_rv["iv_mean"],
        mode="lines",
        name="Daily Average IV"
    ))
    fig_iv_rv.add_trace(go.Scatter(
        x=df_iv_rv["date_time"],
        y=df_iv_rv["rv"],
        mode="lines",
        name="5-Day Rolling Realized Vol (RV)"
    ))
    fig_iv_rv.update_layout(
        title="IV vs RV Comparison (Daily)",
        xaxis_title="Date",
        yaxis_title="Volatility (Decimal Scale)"
    )
    st.plotly_chart(fig_iv_rv, use_container_width=True)
    
    # -------------------------------
    # OPEN INTEREST & DELTA (TICKER) SECTION
    # -------------------------------
    st.subheader("Open Interest & Delta (Options)")
    ticker_list = []
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
        iv = ticker_data.get("iv", None)
        if iv is None:
            continue
        # Compute time to expiry for this instrument using the expiration date from the instrument name
        try:
            expiry_str = instrument.split("-")[1]
            expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
            expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
            now = dt.datetime.now(tz=dt.timezone.utc)
            T_est = (expiry_date - now).total_seconds() / (365 * 24 * 3600)
            if T_est <= 0:
                T_est = 0.0001
        except Exception:
            T_est = 0.05
        S = spot_price
        try:
            d1 = (np.log(S / strike) + 0.5 * iv**2 * T_est) / (iv * np.sqrt(T_est))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": oi,
            "delta": delta_est
        })
    if ticker_list:
        df_ticker = pd.DataFrame(ticker_list)
        fig_bubble = px.scatter(
            df_ticker,
            x="strike",
            y="open_interest",
            size="open_interest",
            color="delta",
            color_continuous_scale=px.colors.diverging.RdBu,
            hover_data=["instrument", "delta"],
            title="Open Interest & Delta for Options"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        tot = df_ticker["open_interest"].sum()
        rat = (df_ticker.apply(lambda r: r["open_interest"] if r["option_type"]=="P" and r["delta"]<0 else 0, axis=1).sum() / tot) if tot else 0
        stat = "Risk-Off" if rat > 0.5 else "Risk-On"
        st.markdown(f"### {stat} (Put OI Ratio: {rat:.2f})")
        fig_indicator = go.Figure(go.Indicator(
            mode="gauge+number+delta", 
            value=rat*100,
            title={'text': "Put OI Ratio (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {'value': 50, 'line': {'color': "black", 'width': 4}, 'thickness': 0.75}
            }
        ))
        fig_indicator.update_layout(height=350, width=450)
        st.plotly_chart(fig_indicator, use_container_width=False)
    else:
        st.warning("No ticker data available for OI & Delta.")
    
    # -------------------------------
    # SEGMENTED ROLLING IV-BASED MARKET REGIME
    # -------------------------------
    st.subheader("Segmented Rolling IV-based Market Regime from 'Optimal Hedge Zone'")
    df_iv_agg = (
        df.groupby("date_time", as_index=False)["iv_close"]
        .mean()
        .rename(columns={"iv_close": "iv_mean"})
    )
    df_iv_agg = df_iv_agg.sort_values("date_time").reset_index(drop=True)
    df_iv_agg["date_time"] = pd.to_datetime(df_iv_agg["date_time"])
    df_iv_agg = df_iv_agg.set_index("date_time")
    df_iv_agg = df_iv_agg.resample("5min").mean().ffill()
    df_iv_agg["iv_rolling_mean"] = df_iv_agg["iv_mean"].rolling("1D").mean()
    df_iv_agg["iv_rolling_std"] = df_iv_agg["iv_mean"].rolling("1D").std()
    df_iv_agg["upper_zone"] = df_iv_agg["iv_rolling_mean"] + df_iv_agg["iv_rolling_std"]
    df_iv_agg["lower_zone"] = df_iv_agg["iv_rolling_mean"] - df_iv_agg["iv_rolling_std"]
    df_iv_agg.dropna(subset=["iv_rolling_mean", "iv_rolling_std", "upper_zone", "lower_zone"], inplace=True)
    
    def label_regime(iv_value, low, high):
        if iv_value > high:
            return "Risk-Off"
        elif iv_value < low:
            return "Risk-On"
        else:
            return "Neutral"
    
    df_iv_agg["market_regime"] = df_iv_agg.apply(
        lambda row: label_regime(row["iv_mean"], row["lower_zone"], row["upper_zone"]),
        axis=1
    )
    df_iv_agg_reset = df_iv_agg.reset_index()
    df_iv_agg_reset["regime_segment"] = (
        df_iv_agg_reset["market_regime"] != df_iv_agg_reset["market_regime"].shift()
    ).cumsum()
    
    fig_iv_regime = go.Figure()
    fig_iv_regime.add_trace(
        go.Scatter(
            x=df_iv_agg_reset["date_time"],
            y=df_iv_agg_reset["upper_zone"],
            mode="lines",
            line=dict(color="gray", dash="dot"),
            name="Upper Zone",
            connectgaps=True
        )
    )
    fig_iv_regime.add_trace(
        go.Scatter(
            x=df_iv_agg_reset["date_time"],
            y=df_iv_agg_reset["lower_zone"],
            mode="lines",
            line=dict(color="gray", dash="dot"),
            name="Lower Zone",
            connectgaps=True
        )
    )
    regime_colors = {"Risk-On": "green", "Risk-Off": "red", "Neutral": "blue"}
    used_regimes = set()
    for seg_id, seg_data in df_iv_agg_reset.groupby("regime_segment"):
        current_regime = seg_data["market_regime"].iloc[0]
        show_legend = current_regime not in used_regimes
        if show_legend:
            used_regimes.add(current_regime)
        fig_iv_regime.add_trace(
            go.Scatter(
                x=seg_data["date_time"],
                y=seg_data["iv_mean"],
                mode="lines",
                name=current_regime,
                line=dict(color=regime_colors.get(current_regime, "gray"), width=3),
                showlegend=show_legend,
                connectgaps=True
            )
        )
    fig_iv_regime.update_layout(
        title="Segmented Rolling IV vs. 'Optimal Hedge Zone' (Market Regime)",
        xaxis_title="Date",
        yaxis_title="Rolling IV"
    )
    st.plotly_chart(fig_iv_regime, use_container_width=True)
    
    # -------------------------------
    # SEGMENTED ROLLING SKEW-BASED MARKET REGIME (CALLS vs. PUTS)
    # -------------------------------
    st.subheader("Segmented Rolling Skew-based Market Regime (Calls vs. Puts)")
    df_calls_mean = (
        df_calls.groupby('date_time', as_index=False)['mark_price_close']
        .mean()
        .rename(columns={'mark_price_close': 'mean_calls'})
    )
    df_puts_mean = (
        df_puts.groupby('date_time', as_index=False)['mark_price_close']
        .mean()
        .rename(columns={'mark_price_close': 'mean_puts'})
    )
    df_mean_diff = pd.merge(df_calls_mean, df_puts_mean, on='date_time', how='outer').sort_values('date_time')
    df_mean_diff['mean_diff'] = df_mean_diff['mean_calls'] - df_mean_diff['mean_puts']
    df_mean_diff['date_time'] = pd.to_datetime(df_mean_diff['date_time'])
    df_mean_diff = df_mean_diff.set_index('date_time').sort_index()
    df_mean_diff = df_mean_diff.resample('5min').mean().ffill()
    df_mean_diff['rolling_mean'] = df_mean_diff['mean_diff'].rolling('1D', min_periods=1).mean()
    df_mean_diff['rolling_std'] = df_mean_diff['mean_diff'].rolling('1D', min_periods=1).std()
    df_mean_diff['upper_zone'] = df_mean_diff['rolling_mean'] + df_mean_diff['rolling_std']
    df_mean_diff['lower_zone'] = df_mean_diff['rolling_mean'] - df_mean_diff['rolling_std']
    df_mean_diff.dropna(subset=['rolling_mean', 'rolling_std', 'upper_zone', 'lower_zone'], inplace=True)
    
    def label_skew_regime(diff_val, low, high):
        if diff_val > high:
            return "Strong Risk-On"
        elif diff_val < low:
            return "Strong Risk-Off"
        else:
            return "Neutral"
    df_mean_diff['skew_regime'] = df_mean_diff.apply(
        lambda row: label_skew_regime(row['mean_diff'], row['lower_zone'], row['upper_zone']),
        axis=1
    )
    df_mean_diff_reset = df_mean_diff.reset_index()
    df_mean_diff_reset['regime_segment'] = (
        df_mean_diff_reset['skew_regime'] != df_mean_diff_reset['skew_regime'].shift()
    ).cumsum()
    
    fig_skew_segments = go.Figure()
    fig_skew_segments.add_trace(
        go.Scatter(
            x=df_mean_diff_reset['date_time'],
            y=df_mean_diff_reset['upper_zone'],
            mode='lines',
            line=dict(color="gray", dash='dot'),
            name='Upper Zone',
            connectgaps=True
        )
    )
    fig_skew_segments.add_trace(
        go.Scatter(
            x=df_mean_diff_reset['date_time'],
            y=df_mean_diff_reset['lower_zone'],
            mode='lines',
            line=dict(color="gray", dash='dot'),
            name='Lower Zone',
            connectgaps=True
        )
    )
    skew_colors = {
        'Strong Risk-On': 'blue',
        'Strong Risk-Off': 'red',
        'Neutral': 'lightblue'
    }
    used_regimes = set()
    for seg_id, seg_data in df_mean_diff_reset.groupby('regime_segment'):
        current_regime = seg_data['skew_regime'].iloc[0]
        show_legend = current_regime not in used_regimes
        if show_legend:
            used_regimes.add(current_regime)
        fig_skew_segments.add_trace(
            go.Scatter(
                x=seg_data['date_time'],
                y=seg_data['mean_diff'],
                mode='lines',
                name=current_regime,
                line=dict(color=skew_colors.get(current_regime, 'gray'), width=3),
                showlegend=show_legend,
                connectgaps=True
            )
        )
    fig_skew_segments.update_layout(
        title="Segmented Rolling Skew-based Market Regime (Calls vs. Puts)",
        xaxis_title="Date/Time",
        yaxis_title="Mean Diff (Calls - Puts)"
    )
    st.plotly_chart(fig_skew_segments, use_container_width=True)
    
    st.subheader("Skew-based Market Regime (Calls vs. Puts)")
    df_calls_mean = (
        df_calls.groupby('date_time', as_index=False)['mark_price_close']
        .mean()
        .rename(columns={'mark_price_close': 'mean_calls'})
    )
    df_puts_mean = (
        df_puts.groupby('date_time', as_index=False)['mark_price_close']
        .mean()
        .rename(columns={'mark_price_close': 'mean_puts'})
    )
    df_mean_diff = pd.merge(df_calls_mean, df_puts_mean, on='date_time', how='outer').sort_values('date_time')
    df_mean_diff['mean_diff'] = df_mean_diff['mean_calls'] - df_mean_diff['mean_puts']
    df_mean_diff['market_regime'] = np.where(
        df_mean_diff['mean_diff'] > 0, 
        'Risk-On', 
        np.where(df_mean_diff['mean_diff'] < 0, 'Risk-Off', 'Neutral')
    )
    fig_skew_regime = px.scatter(
        df_mean_diff,
        x='date_time',
        y='mean_diff',
        color='market_regime',
        title='Skew-based Market Regime (Calls vs. Puts)',
        labels={'mean_diff': 'Mean Calls - Mean Puts'}
    )
    fig_skew_regime.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_skew_regime, use_container_width=True)
    
    # -------------------------------
    # VOLATILITY SMILE & CORRELATION HEATMAPS
    # -------------------------------
    st.subheader("Volatility Smile at Latest Timestamp")
    latest_ts = df["date_time"].max()
    smile_df = df[df["date_time"] == latest_ts]
    if not smile_df.empty:
        atm_strike = smile_df.loc[smile_df["mark_price_close"].idxmax(), "k"]
        smile_df = smile_df.sort_values(by="k")
        fig_vol_smile = px.line(
            smile_df,
            x="k", y="iv_close",
            markers=True,
            title=f"Volatility Smile at {latest_ts.strftime('%d %b %H:%M')}",
            labels={"iv_close": "IV", "k": "Strike"}
        )
        cheap_hedge_strike = smile_df.loc[smile_df["iv_close"].idxmin(), "k"]
        fig_vol_smile.add_vline(
            x=cheap_hedge_strike,
            line=dict(dash="dash", color="green"),
            annotation_text=f"Cheap Hedge ({cheap_hedge_strike})",
            annotation_position="top"
        )
        fig_vol_smile.add_vline(
            x=spot_price,
            line=dict(dash="dash", color="blue"),
            annotation_text=f"Price: {spot_price:.2f}",
            annotation_position="bottom left"
        )
        fig_vol_smile.update_layout(height=400, width=600)
        st.plotly_chart(fig_vol_smile, use_container_width=True)
    
    st.subheader("Mark Price Correlation Between Strikes")
    corr_matrix = df.pivot_table(
        index="date_time", 
        columns="instrument_name", 
        values="mark_price_close"
    ).corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
        title="Mark Price Correlation Between Strikes"
    )
    fig_corr.update_layout(height=800, width=1200)
    st.plotly_chart(fig_corr, use_container_width=False)
    
    st.subheader("Mark Price Evolution by Strike (Heatmap)")
    fig_mark_heatmap = px.density_heatmap(
        df,
        x="date_time", y="k", z="mark_price_close",
        color_continuous_scale="Viridis",
        title="Mark Price Evolution by Strike"
    )
    fig_mark_heatmap.update_layout(height=400, width=800)
    st.plotly_chart(fig_mark_heatmap, use_container_width=True)
    
    # -------------------------------
    # GAMMA & GAMMA EXPOSURE VISUALIZATIONS
    # -------------------------------
    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return
    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")
    df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
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
        row = df_calls[df_calls["instrument_name"] == instrument].iloc[0] if option_type == "C" else df_puts[df_puts["instrument_name"] == instrument].iloc[0]
        gex = compute_gex(row, spot_price, oi)
        gex_data.append({"strike": strike, "gex": gex, "option_type": option_type})
    df_gex = pd.DataFrame(gex_data)
    plot_gex_by_strike(df_gex)
    plot_net_gex(df_gex, spot_price)
    
    # Example usage of the risk adjustment factor function
    risk_factor = compute_risk_adjustment_factor_cf(df_kraken, alpha=0)
    st.write(f"Risk Adjustment Factor (CF): {risk_factor:.2f}")

if __name__ == '__main__':
    main()
