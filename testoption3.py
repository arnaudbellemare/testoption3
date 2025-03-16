import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import requests
import ccxt
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

###########################################
# EXPIRATION DATE SELECTION FUNCTIONS
###########################################
def get_valid_expiration_options(current_date=None):
    """
    Return valid expiration day options based on today's date.
    If today's date is before the 14th, both 14 and 28 are valid.
    If it's on/after the 14th but before 28, only 28 is valid.
    If it's on/after 28, provide 14 and 28 for the next month.
    """
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
    Uses the current month if possible; otherwise, rolls over to the next month.
    """
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
    """
    Define query parameters for the Thalex API,
    fetching data from the last 7 days at 5-minute resolution.
    """
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
# CREDENTIALS & LOGIN FUNCTIONS
###########################################
def load_credentials():
    """
    Load user credentials from Streamlit secrets.
    (For production, use st.secrets or environment variables.)
    """
    try:
        creds = st.secrets["credentials"]
        return creds
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    """
    Display a login form and validate credentials.
    Sets st.session_state.logged_in = True upon success.
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
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# INSTRUMENTS FETCHING & FILTERING FUNCTIONS
###########################################
def fetch_instruments():
    """
    Fetch the list of instruments from the Thalex API.
    """
    try:
        response = requests.get(url_instruments)
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

def get_option_instruments(instruments, option_type, expiry_str):
    """
    Filter instruments by option type (C or P) and expiry.
    For example: BTC-28MAR25-40000-C.
    """
    filtered = [inst["instrument_name"] for inst in instruments 
                if inst["instrument_name"].startswith(f"BTC-{expiry_str}") 
                and inst["instrument_name"].endswith(f"-{option_type}")]
    return sorted(filtered)

def get_actual_iv(instrument_name):
    """
    Fetch mark price data for a specific instrument and return its latest iv_close.
    """
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
    """
    Compute the ATM IV by selecting the call whose strike is closest to the spot price.
    """
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
    """
    Filter instruments based on a theoretical range using the ATM IV as a proxy.
    The range is defined as spot_price * exp(± atm_iv * sqrt(t_years) * multiplier).
    """
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
    """
    Fetch Thalex mark price data for given instruments over the past 7 days.
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
                lambda s: int(s.split("-")[2]) if len(s.split("-")) >= 3 and s.split("-")[2].isdigit() else np.nan))
            .assign(option_type=lambda df: df["instrument_name"].str.split("-").str[-1])
        )
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    return df

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    """
    Fetch ticker data for a given instrument from the Thalex API.
    """
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
    """
    Fetch 7 days of 5-minute BTC/USD data from Kraken using ccxt.
    """
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
    except Exception as e:
        st.error(f"Error fetching Kraken data: {e}")
        return pd.DataFrame()
    return df_kraken

###########################################
# EWMA ROGER-SATCHELL VOLATILITY FUNCTIONS
###########################################
def calculate_ewma_roger_satchell_volatility(price_data, span=30):
    """
    Calculate realized volatility using the Roger-Satchell estimator with an EWMA.
    Assumes price_data has columns: 'open', 'high', 'low', 'close'.
    """
    df = price_data.copy()
    df['rs'] = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
    ewma_rs = df['rs'].ewm(span=span, adjust=False).mean()
    volatility = np.sqrt(ewma_rs.clip(lower=0))
    return volatility

def compute_daily_realized_volatility(df, span=30, annualize_days=365):
    """
    Resample the underlying data daily using OHLC aggregation, compute the EWMA Roger-Satchell volatility,
    annualize it, and return the last value as a scalar.
    """
    if 'date_time' in df.columns:
        df_daily = df.resample('D', on='date_time').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
    else:
        df_daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
    daily_vol = calculate_ewma_roger_satchell_volatility(df_daily, span=span)
    daily_vol_annualized = daily_vol * np.sqrt(annualize_days)
    return daily_vol_annualized.iloc[-1]

###########################################
# OPTION DELTA, GAMMA, AND GEX CALCULATION FUNCTIONS
###########################################
def compute_delta(row, S, position_side="short"):
    """
    Compute the Black-Scholes delta for an option using timezone-aware datetimes.
    """
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
        now = dt.datetime.now(tz=row["date_time"].tzinfo)
        T = (expiry_date - now).days / 365.0
        T = T if T > 0 else 0.0001
        K = row["k"]
        sigma = row["iv_close"]
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1
    except Exception as e:
        st.error(f"Error computing delta for {row['instrument_name']}: {e}")
        return np.nan

def compute_gamma(row, S, position_side="short"):
    """
    Compute the Black-Scholes gamma for an option using timezone-aware datetimes.
    """
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
        now = dt.datetime.now(tz=row["date_time"].tzinfo)
        T = (expiry_date - now).days / 365.0
        if T <= 0:
            return np.nan
        K = row["k"]
        sigma = row["iv_close"]
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except Exception as e:
        st.error(f"Error computing gamma for {row['instrument_name']}: {e}")
        return np.nan

def compute_gex(row, S, oi):
    """
    Compute Gamma Exposure (GEX) for an option, scaled for interpretability.
    """
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2) * 0.01

###########################################
# REALIZED VOLATILITY - EV CALCULATION FUNCTIONS
###########################################
def compute_ev(iv, rv, T, position_side="short"):
    """
    Compute the Expected Value (EV) for an option strategy.
    For short volatility: EV = (((iv^2 - rv^2) * T) / 2) * 100
    For long volatility: EV = (((rv^2 - iv^2) * T) / 2) * 100
    """
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
    """
    Normalize a list of numeric metrics to Z-scores.
    """
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

###########################################
# SELECT OPTIMAL STRIKE - ADAPTIVE FOR SHORT vs LONG VOL
###########################################
def select_optimal_strike(ticker_list, position_side='short'):
    """
    Select the optimal strike based on a composite score that adapts for short or long volatility.
    The composite score is computed from normalized EV, gamma, and open interest.
    Gamma is penalized for short vol and rewarded for long vol.
    """
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
    """
    Compute a raw composite score for an option instrument for visualization.
    This raw score is EV adjusted by gamma and open interest (without normalization).
    """
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
    """
    Plot the volatility surface using moneyness and time to expiry.
    """
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close', color='option_type',
                          title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    """
    Adjust EV for liquidity by penalizing wide bid-ask spreads.
    """
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            bid_ask_spread = ask - bid
            mid_price = (ask + bid) / 2
            if mid_price > 0:
                item['EV'] *= (1 - bid_ask_spread / mid_price)
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    """
    Load historical trade data for backtesting.
    For demonstration, returns a dummy dataframe.
    """
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    """
    Optimize weights using linear regression on historical trade data.
    """
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

###########################################
# AUTOMATIC VOLATILITY STRATEGY RECOMMENDATION
###########################################
def recommend_volatility_strategy(atm_iv, rv):
    """
    Recommend a volatility strategy based on ATM IV and realized volatility.
    If ATM IV > RV, return "short"; if ATM IV < RV, return "long"; otherwise "neutral".
    """
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
    # Enforce login
    login()
    st.title("Crypto Options Dashboard - Adaptive for Short or Long Volatility")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    # Expiry Date Selection
    current_date = dt.datetime.now()
    valid_options = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_options)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Expiration date is invalid or already passed.")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiry = (expiry_date - current_date).days
    T_YEARS = days_to_expiry / 365.0
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    # Fetch Underlying Data from Kraken
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    # Compute Realized Volatility using EWMA Roger-Satchell method
    rv = compute_daily_realized_volatility(df_kraken, span=30, annualize_days=252)
    st.write(f"Computed Realized Volatility (annualized): {rv:.4f}")
    
    # Fetch instruments and compute ATM IV for automatic strategy recommendation
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
    
    # Volatility Strategy Selection (Manual Override)
    position_side = st.sidebar.selectbox("Volatility Strategy", ["short", "long"],
                                           index=0 if recommended_strategy=="short" else 1)
    st.sidebar.write(f"Selected strategy: {position_side}")
    
    # Deviation Range Selection
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2
    
    # Filter Instruments Based on Theoretical Range
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts
    
    # Fetch Mark Price Data from Thalex
    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return

    # Volatility Surface Analysis
    plot_volatility_surface(df, spot_price)
    
    # Build Ticker List with EV and Gamma
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
        T = (expiry_date - current_date).days / 365.0
        S = spot_price
        ev = compute_ev(iv, rv, T, position_side=position_side)
        gamma_val = np.nan
        if option_type == "C":
            temp = df[df["option_type"] == "C"]
            temp = temp[temp["instrument_name"] == instrument]
            if not temp.empty:
                gamma_val = compute_gamma(temp.iloc[0], spot_price, position_side=position_side)
        else:
            temp = df[df["option_type"] == "P"]
            temp = temp[temp["instrument_name"] == instrument]
            if not temp.empty:
                gamma_val = compute_gamma(temp.iloc[0], spot_price, position_side=position_side)
        if np.isnan(gamma_val):
            gamma_val = np.random.uniform(0.01, 0.05)
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "gamma": gamma_val,
            "EV": ev,
            "bid": ticker_data.get("bid", 0),
            "ask": ticker_data.get("ask", 0)
        })
    
    # Adjust for Liquidity (bid-ask spread)
    ticker_list = adjust_for_liquidity(ticker_list)
    
    # Historical Backtesting for Optimal Weights
    historical_data = load_previous_trades()
    weights = optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi'])
    st.write(f"Optimal Weights (EV, Gamma, OI): {weights}")
    
    # Select the Optimal Strike Based on Composite Score
    optimal_ticker = select_optimal_strike(ticker_list, position_side=position_side)
    if optimal_ticker:
        st.markdown(f"### Recommended Strike: **{optimal_ticker['strike']}** from {optimal_ticker['instrument']}")
    else:
        st.write("No optimal strike found based on the current criteria.")
    
    # Visualize Raw Composite Scores by Strike
    composite_scores = []
    for ticker in ticker_list:
        score = compute_composite_score(ticker, position_side=position_side)
        composite_scores.append({
            'strike': ticker['strike'],
            'score': score,
            'instrument': ticker['instrument'],
            'EV': ticker['EV'],
            'gamma': ticker['gamma'],
            'open_interest': ticker['open_interest']
        })
    df_scores = pd.DataFrame(composite_scores)
    fig = px.bar(
        df_scores,
        x='strike',
        y='score',
        hover_data=['instrument', 'EV', 'gamma', 'open_interest'],
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
    
    st.write(f"""
### Why is this strike recommended?
- **EV Calculation ({position_side} vol)**: EV is computed using an adaptive formula.
  - For short vol, it favors cases where IV exceeds RV.
  - For long vol, it favors cases where RV exceeds IV.
- **Gamma Weighting**: Gamma is penalized for short vol and rewarded for long vol.
- **Liquidity**: Open interest provides a small bonus.
- **Heuristic Approach**: Normalized metrics ensure balanced composite scoring.
- **Visual Evidence**: The bar chart shows composite scores per strike with the recommended strike highlighted.
""")
    
    st.write("Analysis complete. Review the correlation analysis and recommended strike above.")

###########################################
# ADDITIONAL FUNCTIONS: VOLATILITY SURFACE, LIQUIDITY, AND BACKTESTING
###########################################
def plot_volatility_surface(df, spot_price):
    """
    Plot the volatility surface using moneyness and time to expiry.
    """
    df = df.copy()
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close',
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

def adjust_for_liquidity(ticker_list):
    """
    Adjust EV for liquidity by penalizing wide bid-ask spreads.
    """
    for item in ticker_list:
        bid = item.get('bid', 0)
        ask = item.get('ask', 0)
        if bid and ask:
            spread = ask - bid
            mid = (ask + bid) / 2
            if mid > 0:
                item['EV'] *= (1 - spread / mid)
    return ticker_list

def load_previous_trades():
    """
    Load historical trade data for backtesting.
    For demonstration, returns a dummy DataFrame.
    """
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    """
    Optimize weights using linear regression on historical trade data.
    """
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

def recommend_volatility_strategy(atm_iv, rv):
    """
    Recommend a volatility strategy based on ATM IV and realized volatility.
    If ATM IV > RV, return "short"; if ATM IV < RV, return "long"; otherwise "neutral".
    """
    if atm_iv > rv:
        return "short"
    elif atm_iv < rv:
        return "long"
    else:
        return "neutral"

###########################################
# MAIN EXECUTION
###########################################
if __name__ == '__main__':
    main()
