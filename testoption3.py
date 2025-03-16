import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import requests
import ccxt
from toolz.curried import *
import plotly.express as px
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
    """Compute the expiration date based on the selected day."""
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
    """Define query parameters for the Thalex API."""
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
    """Load user credentials from Streamlit secrets."""
    try:
        creds = st.secrets["credentials"]
        return creds
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
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# INSTRUMENTS FETCHING & FILTERING FUNCTIONS
###########################################
def fetch_instruments():
    """Fetch the list of instruments from the Thalex API."""
    try:
        response = requests.get(url_instruments)
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

def get_option_instruments(instruments, option_type, expiry_str):
    """Filter instruments by option type and expiry."""
    filtered = [inst["instrument_name"] for inst in instruments 
                if inst["instrument_name"].startswith(f"BTC-{expiry_str}") 
                and inst["instrument_name"].endswith(f"-{option_type}")]
    return sorted(filtered)

def get_atm_iv(calls_all, spot_price):
    """Compute the ATM IV by selecting the call closest to the spot price."""
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
    """Filter instruments based on a theoretical range using ATM IV."""
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
    """Fetch Thalex mark price data for given instruments."""
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
    """Fetch ticker data for a given instrument from the Thalex API."""
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
    """Fetch 7 days of 5-minute BTC/USD data from Kraken using ccxt."""
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
# VOLATILITY SURFACE ANALYSIS
###########################################
def plot_volatility_surface(df, spot_price):
    """Plot the volatility surface using moneyness and time to expiry."""
    df['moneyness'] = df['k'] / spot_price
    df['T'] = (df['date_time'].max() - df['date_time']).dt.days / 365.0
    fig = px.scatter_3d(df, x='moneyness', y='T', z='iv_close', 
                        color='option_type', title="Volatility Surface")
    st.plotly_chart(fig)

###########################################
# TRANSACTION COST ADJUSTMENT
###########################################
def adjust_for_liquidity(ticker_list):
    """Adjust EV for bid-ask spreads."""
    for item in ticker_list:
        bid_ask_spread = item.get('ask', 0) - item.get('bid', 0)
        mid_price = (item.get('ask', 0) + item.get('bid', 0)) / 2
        if mid_price > 0:
            item['EV'] *= (1 - bid_ask_spread / mid_price)  # Penalize wide spreads
    return ticker_list

###########################################
# HISTORICAL BACKTESTING FOR OPTIMAL WEIGHTS
###########################################
def load_previous_trades():
    """Load historical trade data for backtesting."""
    # Replace with actual historical data loading logic
    return pd.DataFrame({
        'EV': np.random.normal(0, 1, 100),
        'gamma': np.random.normal(0, 1, 100),
        'oi': np.random.normal(0, 1, 100),
        'profit': np.random.normal(0, 1, 100)
    })

def optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi']):
    """Optimize weights using linear regression."""
    X = historical_data[features]
    y = historical_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_

###########################################
# MAIN DASHBOARD FUNCTION
###########################################
def main():
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
    
    # Volatility Strategy Selection
    position_side = st.sidebar.selectbox("Volatility Strategy", ["short", "long"])
    st.sidebar.write(f"Selected strategy: {position_side}")
    
    # Deviation Range Selection
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2

    # Fetch Underlying Data from Kraken
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")

    # Compute Realized Volatility
    rv = compute_daily_realized_volatility(df_kraken, span=30, annualize_days=252)
    st.write(f"Computed Realized Volatility (annualized): {rv:.4f}")

    # Filter Instruments
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts

    # Fetch Mark Price Data
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
            strike = int(instrument.split("-")[2]))
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        iv = ticker_data.get("iv", None)
        if iv is None:
            continue
        T = (expiry_date - current_date).days / 365.0
        S = spot_price
        
        # Compute EV
        ev = compute_ev(iv, rv, T, position_side=position_side)
        
        # Compute Gamma
        gamma_val = np.nan
        if option_type == "C":
            temp = df_calls[df_calls["instrument_name"] == instrument]
            if not temp.empty:
                gamma_val = compute_gamma(temp.iloc[0], spot_price, position_side=position_side)
        else:
            temp = df_puts[df_puts["instrument_name"] == instrument]
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
    
    # Adjust for Liquidity
    ticker_list = adjust_for_liquidity(ticker_list)

    # Historical Backtesting for Optimal Weights
    historical_data = load_previous_trades()
    weights = optimize_weights(historical_data, target='profit', features=['EV', 'gamma', 'oi'])
    st.write(f"Optimal Weights (EV, Gamma, OI): {weights}")

    # Select Optimal Strike
    optimal_ticker = select_optimal_strike(ticker_list, position_side=position_side)
    if optimal_ticker:
        st.markdown(f"### Recommended Strike: **{optimal_ticker['strike']}** from {optimal_ticker['instrument']}")
    else:
        st.write("No optimal strike found based on the current criteria.")

if __name__ == '__main__':
    main()
