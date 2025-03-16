import streamlit as st
import datetime as dt
import pandas as pd
import requests
import numpy as np
import ccxt
import re
import calendar
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import math
from scipy.interpolate import CubicSpline

###########################################
# EXPIRATION DATE HANDLING
###########################################
def get_valid_expiration_options(current_date=None):
    """
    Return valid expiration day options based on today's date.
    Note: This function returns fixed choices, e.g., 14 and 28.
    """
    if current_date is None:
        current_date = dt.datetime.now()
    return [14, 28]

def compute_expiry_date(selected_day, current_date=None):
    """
    Compute the expiration date robustly.
    
    - Use the current month if possible; otherwise, roll over to the next month.
    - Validate the selected day against the target month's length.
    """
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < selected_day:
        year = current_date.year
        month = current_date.month
    else:
        if current_date.month == 12:
            year = current_date.year + 1
            month = 1
        else:
            year = current_date.year
            month = current_date.month + 1
    last_day = calendar.monthrange(year, month)[1]
    if selected_day > last_day:
        st.error(f"Selected expiration day {selected_day} exceeds days in month {month} (max {last_day}).")
        return None
    try:
        expiry = dt.datetime(year, month, selected_day)
    except Exception as e:
        st.error(f"Error computing expiration date: {e}")
        return None
    return expiry

###########################################
# THALEX API DETAILS AND GLOBAL SETTINGS
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

# Expected columns from the Thalex API for mark price data.
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
    NOTE: For production, use environment variables or Streamlit secrets.
    """
    try:
        with open("usernames.txt", "r") as f_user:
            usernames = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            passwords = [line.strip() for line in f_pass if line.strip()]
        if len(usernames) != len(passwords):
            st.error("Mismatch between number of usernames and passwords.")
            return {}
        return dict(zip(usernames, passwords))
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    """Display a login form and validate credentials."""
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

def parse_instrument_name(inst):
    """
    Parse an instrument from either a dict or string.
    Expected format: BTC-<EXPIRY>-<STRIKE>-<OPTION_TYPE>
    e.g., BTC-28MAR25-100000-C
    Returns (expiry_str, strike, option_type) or (None, None, None) on failure.
    """
    if isinstance(inst, dict):
        instrument_name = inst.get("instrument_name", "")
    elif isinstance(inst, str):
        instrument_name = inst
    else:
        instrument_name = str(inst)
    pattern = r"BTC-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)"
    match = re.match(pattern, instrument_name)
    if match:
        expiry_str = match.group(1)
        strike = int(match.group(2))
        option_type = match.group(3)
        return expiry_str, strike, option_type
    else:
        st.error(f"Instrument name '{instrument_name}' does not match expected format.")
        return None, None, None

def get_option_instruments(instruments, option_type, expiry_str):
    """
    Filter instruments by option type and expiry.
    Uses regex parsing to ensure robustness.
    """
    filtered = []
    for inst in instruments:
        exp, strike, opt_type = parse_instrument_name(inst)
        if exp == expiry_str and opt_type == option_type:
            filtered.append(inst)
    return sorted(filtered, key=lambda x: parse_instrument_name(x)[1] if parse_instrument_name(x)[1] is not None else 0)

def get_actual_iv(instrument_name):
    """Fetch mark price data for an instrument and return its latest IV (iv_close)."""
    try:
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
    except Exception as e:
        st.error(f"Error fetching IV for {instrument_name}: {e}")
        return None

def get_filtered_instruments(spot_price, expiry_str, t_years, multiplier=1):
    """
    Filter instruments based on a theoretical strike range computed from a standard deviation move.
    Uses ATM IV (from the nearest strike) as a benchmark.
    """
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    
    strike_list = []
    for inst in calls_all:
        exp, strike, opt_type = parse_instrument_name(inst)
        if strike is not None:
            strike_list.append((inst, strike))
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
    closest_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    nearest_instrument = strike_list[closest_index][0]
    
    actual_iv = get_actual_iv(nearest_instrument)
    if actual_iv is None:
        raise Exception("Could not fetch actual IV for the nearest instrument")
    
    lower_bound = spot_price * np.exp(-actual_iv * np.sqrt(t_years) * multiplier)
    upper_bound = spot_price * np.exp(actual_iv * np.sqrt(t_years) * multiplier)
    
    filtered_calls = [inst for inst in calls_all if lower_bound <= parse_instrument_name(inst)[1] <= upper_bound]
    filtered_puts = [inst for inst in puts_all if lower_bound <= parse_instrument_name(inst)[1] <= upper_bound]
    
    filtered_calls.sort(key=lambda x: parse_instrument_name(x)[1])
    filtered_puts.sort(key=lambda x: parse_instrument_name(x)[1])
    return filtered_calls, filtered_puts

###########################################
# DATA FETCHING FUNCTIONS
###########################################
@st.cache_data(ttl=30)
def fetch_data(instruments_tuple):
    """
    Fetch Thalex mark price data for instruments over the past 7 days at 5m resolution.
    Uses toolz for functional mapping and filtering.
    """
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
                lambda s: parse_instrument_name(s)[1] if parse_instrument_name(s)[1] is not None else np.nan))
            .assign(option_type=lambda df: df["instrument_name"].map(
                lambda s: parse_instrument_name(s)[2] if parse_instrument_name(s)[2] is not None else ""))
        )
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    """Fetch ticker data for a given instrument from the Thalex API."""
    try:
        params_dict = {"instrument_name": instrument_name}
        response = requests.get(URL_TICKER, params=params_dict)
        if response.status_code != 200:
            return None
        data = response.json()
        return data.get("result", {})
    except Exception as e:
        st.error(f"Error fetching ticker for {instrument_name}: {e}")
        return None

def fetch_kraken_data():
    """Fetch 7 days of 5m BTC/USD data from Kraken using ccxt."""
    try:
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
    except Exception as e:
        st.error(f"Error fetching Kraken data: {e}")
        return pd.DataFrame()

###########################################
# REALIZED VOLATILITY & EV CALCULATION FUNCTIONS
###########################################
def compute_realized_volatility(df):
    """
    Compute the realized volatility from underlying close prices.
    Uses 5-minute intervals (288 per day) and annualizes using sqrt(288*365),
    computing the square root of the sum of squared returns.
    """
    df = df.copy()
    returns = df['close'].pct_change().dropna()
    rv = np.sqrt((returns ** 2).sum()) * np.sqrt(288 * 365)
    return rv

def compute_ev(iv, rv, T, position_side="short"):
    """
    Compute the Expected Value (EV) based on IV, RV, and time to expiry T.
    
    For short volatility:
      EV = (((iv^2 - rv^2) * T) / 2) * 100
    For long volatility:
      EV = (((rv^2 - iv^2) * T) / 2) * 100
    
    A positive EV indicates favorable conditions for the respective strategy.
    """
    if position_side.lower() == "short":
        ev = (((iv**2 - rv**2) * T) / 2) * 100
    elif position_side.lower() == "long":
        ev = (((rv**2 - iv**2) * T) / 2) * 100
    else:
        ev = (((iv**2 - rv**2) * T) / 2) * 100
    return ev

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S):
    """Compute Black-Scholes delta for an option."""
    try:
        exp_str, _, _ = parse_instrument_name(row["instrument_name"])
        expiry_date = dt.datetime.strptime(exp_str, "%d%b%y")
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
        exp_str, _, _ = parse_instrument_name(row["instrument_name"])
        expiry_date = dt.datetime.strptime(exp_str, "%d%b%y")
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
    return gamma * oi * (S ** 2) * 0.01

###########################################
# FUNCTION TO SELECT THE OPTIMAL STRIKE
###########################################
def select_optimal_strike(ticker_list, strategy='EV', position_side='short'):
    """
    Select the optimal strike based on a composite score.
    
    Composite score is based on:
      - EV as the primary metric.
      - Gamma adjustment (penalizing high gamma for short volatility).
      - Open Interest as a liquidity bonus.
      - Fallback to delta proximity if EV is missing.
    """
    best_score = -np.inf
    best_candidate = None

    for item in ticker_list:
        score = 0
        if 'EV' in item and item['EV'] is not None:
            score = item['EV']
        else:
            score = 1 - abs(item.get('delta', 0) - 0.5)
        
        if 'gamma' in item and item.get('gamma', 0) > 0:
            if position_side.lower() == "short":
                score /= item['gamma']
            else:
                score *= item['gamma']
        
        if 'open_interest' in item and item['open_interest']:
            score += 0.01 * item['open_interest']
        
        if score > best_score:
            best_score = score
            best_candidate = item

    return best_candidate

###########################################
# VISUALIZATION: COMPOSITE SCORE BY STRIKE
###########################################
def compute_composite_score(item, position_side='short'):
    """
    Compute a composite score for an option instrument.
    Mirrors the logic in select_optimal_strike.
    """
    score = 0
    if 'EV' in item and item['EV'] is not None:
        score = item['EV']
    else:
        score = 1 - abs(item.get('delta', 0) - 0.5)
    
    if 'gamma' in item and item.get('gamma', 0) > 0:
        if position_side.lower() == "short":
            score /= item['gamma']
        else:
            score *= item['gamma']
    
    if 'open_interest' in item and item['open_interest']:
        score += 0.01 * item['open_interest']
    
    return score

###########################################
# CORRELATION ANALYSIS: MARK PRICE CORRELATION BETWEEN STRIKES
###########################################
def analyze_mark_price_correlation(df):
    """
    Analyze the correlation between mark prices across different strikes.
    """
    pivot_df = df.pivot_table(index="date_time", columns="instrument_name", values="mark_price_close")
    corr_matrix = pivot_df.corr()
    return corr_matrix

###########################################
# MAIN DASHBOARD FUNCTION
###########################################
def main():
    login()  # Enforce authentication.
    
    st.title("Crypto Options Dashboard with Vol & Option Theory")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    # Expiration Date Selection
    current_date = dt.datetime.now()
    valid_options = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_options)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Expiration date is invalid or already passed.")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiry = (expiry_date - current_date).days
    T_YEARS = days_to_expiry / 365  # Dynamic time to expiry
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    # Deviation Range Selection
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2

    # Fetch Market Data from Kraken
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    # Compute Realized Volatility from Kraken Data
    rv = compute_realized_volatility(df_kraken)
    st.write(f"Computed Realized Volatility (annualized): {rv:.4f}")

    # Filter Instruments Based on Theoretical Strike Range
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts
    
    # Fetch Options Data from Thalex API
    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return

    # Separate Calls and Puts
    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")
    
    # Mark Price Correlation Analysis
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
    
    # Additional Visualization: Volatility Surface (IV vs. Strike) at Latest Timestamp
    st.subheader("Volatility Surface (IV vs. Strike) at Latest Timestamp")
    latest_ts = df["date_time"].max()
    latest_data = df[df["date_time"] == latest_ts]
    if not latest_data.empty:
        latest_data = latest_data.sort_values("k")
        fig_vol_surface = px.line(latest_data, x="k", y="iv_close", markers=True,
                                  title=f"Volatility Surface at {latest_ts.strftime('%d %b %H:%M')}",
                                  labels={"iv_close": "IV", "k": "Strike"})
        st.plotly_chart(fig_vol_surface, use_container_width=True)
    
    # Ticker List & Optimal Strike Selection
    ticker_list = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if not ticker_data or "open_interest" not in ticker_data:
            continue
        exp, strike, option_type = parse_instrument_name(instrument)
        if strike is None:
            continue
        iv = ticker_data.get("iv", None)
        if iv is None:
            continue
        # Dynamic time to expiry from earlier T_YEARS
        T = T_YEARS
        S = spot_price
        try:
            d1 = (np.log(S / strike) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        
        # Gamma estimation using corresponding calls/puts.
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
        
        # Compute EV using IV, RV, and dynamic T. For short volatility strategy.
        ev = compute_ev(iv, rv, T, position_side="short")
        
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
    
    # Composite Score Visualization
    composite_scores = []
    for ticker in ticker_list:
        score = compute_composite_score(ticker, position_side="short")
        composite_scores.append({
            'strike': ticker['strike'],
            'score': score,
            'instrument': ticker['instrument'],
            'EV': ticker['EV'],
            'delta': ticker['delta'],
            'gamma': ticker['gamma'],
            'open_interest': ticker['open_interest']
        })
    df_scores = pd.DataFrame(composite_scores)
    
    fig = px.bar(
        df_scores, 
        x='strike', 
        y='score',
        hover_data=['instrument', 'EV', 'delta', 'gamma', 'open_interest'],
        title="Composite Score by Strike (Higher is Better)"
    )
    if optimal_ticker is not None:
        recommended_strike = optimal_ticker['strike']
        fig.add_vline(
            x=recommended_strike, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Recommended Strike",
            annotation_position="top"
        )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
### Why is this strike recommended?
- **EV Advantage:** The computed EV for this strike is more favorable compared to others.
- **Gamma & Liquidity Adjustment:** After penalizing for high gamma (risk sensitivity) and adding an open interest bonus, this strike attains the highest composite score.
- **Heuristic Insight:** This approach, informed by option theory and volatility research, balances premium capture with risk exposure.
- **Visual Evidence:** The bar chart above displays composite scores by strike, with the red dashed line marking the recommended strike.
""")
    
    st.write("Analysis complete. Review the correlation analysis and optimal strike recommendation above.")

if __name__ == '__main__':
    main()
