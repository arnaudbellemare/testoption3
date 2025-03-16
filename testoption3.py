import streamlit as st
import datetime as dt
import pandas as pd
import requests
import numpy as np
import ccxt
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, percentileofscore
import math
from scipy.interpolate import CubicSpline

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
    Uses current month if possible; otherwise, rolls over to next month.
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
# Thalex API details and global settings
###########################################
BASE_URL = "https://thalex.com/api/v2/public"
instruments_endpoint = "instruments"  # For fetching available instruments
url_instruments = f"{BASE_URL}/{instruments_endpoint}"
mark_price_endpoint = "mark_price_historical_data"
url_mark_price = f"{BASE_URL}/{mark_price_endpoint}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"
windows = {"7D": "vrp_7d"}  # Example rolling window configuration

def params(instrument_name):
    now = dt.datetime.now()
    start_dt = now - dt.timedelta(days=7)
    return {
        "from": int(start_dt.timestamp()),
        "to": int(now.timestamp()),
        "resolution": "5m",
        "instrument_name": instrument_name,
    }

# Expected column names from the Thalex API for mark price data.
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
    """Load user credentials from text files."""
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
    """Displays a login form and validates credentials."""
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
                st.success("Logged in successfully!")
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# INSTRUMENTS FETCHING & FILTERING FUNCTIONS
###########################################
def fetch_instruments():
    """Fetch instruments list from the Thalex API."""
    response = requests.get(url_instruments)
    if response.status_code != 200:
        raise Exception("Failed to fetch instruments")
    data = response.json()
    return data.get("result", [])

def get_option_instruments(instruments, option_type, expiry_str):
    """
    Filter instruments by option type (C or P) and expiry.
    Assumes instrument naming convention includes expiry and strike.
    """
    filtered = [
        inst["instrument_name"] for inst in instruments
        if inst["instrument_name"].startswith(f"BTC-{expiry_str}") and inst["instrument_name"].endswith(f"-{option_type}")
    ]
    return sorted(filtered)

def get_actual_iv(instrument_name):
    """Fetch mark price data for an instrument and return its latest IV (iv_close)."""
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
    Filter instruments based on a theoretical range computed from a standard deviation move.
    Uses the nearest instrument to derive an actual IV and then applies bounds.
    """
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    
    # Critical assumption: instrument naming includes strike as the third segment.
    strike_list = [(inst, int(inst.split("-")[2])) for inst in calls_all]
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
    closest_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    nearest_instrument = strike_list[closest_index][0]
    
    actual_iv = get_actual_iv(nearest_instrument)
    if actual_iv is None:
        raise Exception("Could not fetch actual IV for the nearest instrument")
    
    # Compute theoretical bounds using exponential scaling.
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
    Fetch Thalex mark price data for instruments over the past 7 days at 5m resolution.
    Uses functional programming (toolz) for mapping and filtering.
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
        .assign(k=lambda df: df["instrument_name"].map(
            lambda s: int(s.split("-")[2]) if len(s.split("-")) >= 3 and s.split("-")[2].isdigit() else np.nan))
        .assign(option_type=lambda df: df["instrument_name"].str.split("-").str[-1])
    )
    return df

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    """Fetch ticker data for a given instrument from the Thalex API."""
    params_dict = {"instrument_name": instrument_name}
    response = requests.get(URL_TICKER, params=params_dict)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("result", {})

def fetch_kraken_data():
    """Fetch 7 days of 5m BTC/USD data from Kraken using ccxt."""
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
# REALIZED VOLATILITY & EV CALCULATION FUNCTIONS
###########################################
def compute_realized_volatility(df):
    """
    Compute the realized volatility from the underlying's close prices.
    Assumes 5-minute intervals; annualizes using sqrt(288 * 365).
    """
    df = df.copy()
    df['return'] = df['close'].pct_change()
    # 5-minute intervals: 288 per day; annualize with sqrt(288*365)
    rv = df['return'].std() * np.sqrt(288 * 365)
    return rv

def compute_ev(iv, rv, T, position_side="short"):
    """
    Compute the Expected Value (EV) based on implied volatility, realized volatility, and time to expiry.
    
    For short volatility:
      EV = (((iv^2 - rv^2) * T) / 2) * 100
      
    For long volatility:
      EV = (((rv^2 - iv^2) * T) / 2) * 100
      
    This ensures a positive EV indicates a favorable condition for the respective strategy.
    """
    if position_side.lower() == "short":
        ev = (((iv**2 - rv**2) * T) / 2) * 100
    elif position_side.lower() == "long":
        ev = (((rv**2 - iv**2) * T) / 2) * 100
    else:
        ev = (((iv**2 - rv**2) * T) / 2) * 100  # default to short volatility calculation
    return ev

###########################################
# OPTION DELTA, GAMMA, and GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    """Compute Black-Scholes delta for an option."""
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    T = T if T > 0 else 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    if sigma <= 0:
        return np.nan
    try:
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

def compute_gamma(row, S):
    """Compute Black-Scholes gamma for an option."""
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
    if sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def compute_gex(row, S, oi):
    """Compute Gamma Exposure (GEX) for an option."""
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    # Scale for interpretability; note sensitivity to underlying and gamma.
    return gamma * oi * (S ** 2) * 0.01

###########################################
# FUNCTION TO SELECT THE OPTIMAL STRIKE
###########################################
def select_optimal_strike(ticker_list, strategy='EV', position_side='short'):
    """
    Select the optimal strike based on a composite score.
    
    Composite Score Components:
      - Expected Value (EV) is the primary metric.
      - Gamma adjustment: Penalize high gamma for short volatility.
      - Open Interest: Add a liquidity bonus.
      - Fallback to delta proximity (prefer near-the-money).
    
    This function reflects recommendations from research and expert insights.
    """
    best_score = -np.inf
    best_candidate = None

    for item in ticker_list:
        score = 0
        # Use computed EV as primary metric.
        if 'EV' in item and item['EV'] is not None:
            score = item['EV']
            if position_side.lower() == "long":
                score = score  # Already computed for long vol
        else:
            # Fallback: Favor near-the-money options.
            score = 1 - abs(item.get('delta', 0) - 0.5)
        
        # Gamma adjustment.
        if 'gamma' in item and item.get('gamma', 0) > 0:
            if position_side.lower() == "short":
                score /= item['gamma']
            else:
                score *= item['gamma']
        
        # Incorporate open interest as a liquidity bonus.
        if 'open_interest' in item and item['open_interest']:
            score += 0.01 * item['open_interest']
        
        if score > best_score:
            best_score = score
            best_candidate = item

    return best_candidate

###########################################
# CORRELATION ANALYSIS: MARK PRICE CORRELATION BETWEEN STRIKES
###########################################
def analyze_mark_price_correlation(df):
    """
    Analyze the correlation between mark prices across different strikes.
    
    Key Insights:
      - High correlation indicates consistent pricing and proper model calibration.
      - Anomalies may signal liquidity issues or model biases.
      - Research on volatility surfaces emphasizes smooth correlation patterns.
    
    Returns:
      A correlation matrix DataFrame.
    """
    pivot_df = df.pivot_table(index="date_time", columns="instrument_name", values="mark_price_close")
    corr_matrix = pivot_df.corr()
    return corr_matrix

###########################################
# MAIN DASHBOARD FUNCTION
###########################################
def main():
    login()  # Enforce authentication.
    
    st.title("Crypto Options Dashboard with Mark Price Correlation Analysis")
    
    # Optional Logout
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    # EXPIRATION DATE SELECTION
    current_date = dt.datetime.now()
    valid_options = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_options)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Expiration date is invalid or already passed.")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiry = (expiry_date - current_date).days
    T_YEARS = days_to_expiry / 365  # Time to expiry in years.
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    # DEVIATION RANGE SELECTION
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2

    # Fetch market data from Kraken.
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    # Compute realized volatility from Kraken data.
    rv = compute_realized_volatility(df_kraken)
    st.write(f"Computed Realized Volatility (annualized): {rv:.4f}")

    # Filter instruments based on theoretical strike range.
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts
    
    # Fetch options data from Thalex API.
    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return

    # Separate calls and puts.
    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")
    
    # -------------------- Mark Price Correlation Analysis --------------------
    st.subheader("Mark Price Correlation Between Strikes")
    corr_matrix = analyze_mark_price_correlation(df)
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
    
    # -------------------- Additional Visualizations --------------------
    st.subheader("Mark Price Evolution by Strike (Heatmap)")
    fig_mark_heatmap = px.density_heatmap(
        df,
        x="date_time", y="k", z="mark_price_close",
        color_continuous_scale="Viridis",
        title="Mark Price Evolution by Strike"
    )
    fig_mark_heatmap.update_layout(height=400, width=800)
    st.plotly_chart(fig_mark_heatmap, use_container_width=True)
    
    # -------------------- Ticker List & Optimal Strike Selection --------------------
    ticker_list = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if not ticker_data or "open_interest" not in ticker_data:
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        iv = ticker_data.get("iv", None)
        if iv is None:
            continue
        T_est = 0.05  # Placeholder for time to expiry; adjust as needed.
        S = spot_price
        try:
            d1 = (np.log(S / strike) + 0.5 * iv**2 * T_est) / (iv * np.sqrt(T_est))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        
        # Gamma estimation.
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
            gamma_val = np.random.uniform(0.01, 0.05)
        
        # Compute EV using the computed realized volatility.
        # The EV calculation now adjusts based on the volatility view (short or long).
        ev = compute_ev(iv, rv, T_YEARS, position_side="short")
        
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "delta": delta_est,
            "gamma": gamma_val,
            "EV": ev
        })
    
    optimal_ticker = select_optimal_strike(ticker_list, strategy='EV', position_side='short')
    if optimal_ticker:
        st.markdown(f"### Recommended Strike: **{optimal_ticker['strike']}** from {optimal_ticker['instrument']}")
    else:
        st.write("No optimal strike found based on the current criteria.")
    
    st.write("Analysis complete. Review the correlation analysis and optimal strike recommendation above.")

if __name__ == '__main__':
    main()
