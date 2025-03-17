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
# Global Settings
###########################################
BASE_URL = "https://thalex.com/api/v2/public"
instruments_endpoint = "instruments"  # Endpoint for fetching available instruments
url_instruments = f"{BASE_URL}/{instruments_endpoint}"
mark_price_endpoint = "mark_price_historical_data"
url_mark_price = f"{BASE_URL}/{mark_price_endpoint}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"

DEFAULT_EXPIRY_STR = "28MAR25"  # Expected format: %d%b%y
expiry_date = dt.datetime.strptime(DEFAULT_EXPIRY_STR, "%d%b%y")
current_date = dt.datetime.now()
days_to_expiry = (expiry_date - current_date).days
T_YEARS = days_to_expiry / 365

windows = {"7D": "vrp_7d"}

###########################################
# Helper Functions
###########################################
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
# Risk Adjustment using Cornish-Fisher
###########################################
def compute_risk_adjustment_factor_cf(df, alpha=0.05):
    df = df.copy()
    if "close" not in df.columns:
        return 1.0
    returns = df['close'].pct_change().dropna()
    S = returns.skew()
    K = returns.kurtosis() + 3  
    z = norm.ppf(alpha)
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * (K - 3) / 24 - (2*z**3 - 5*z) * (S**2) / 36
    risk_factor = abs(z_cf / z) if z != 0 else 1.0
    return risk_factor

###########################################
# Expiration Date Helpers
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
    except ValueError:
        st.error("Invalid expiration date.")
        return None
    return expiry

###########################################
# Credentials & Login Functions
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
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# Instruments Fetching & Filtering Functions
###########################################
def fetch_instruments():
    response = requests.get(url_instruments)
    if response.status_code != 200:
        raise Exception("Failed to fetch instruments")
    data = response.json()
    return data.get("result", [])

def get_option_instruments(instruments, option_type, expiry_str):
    return sorted([inst["instrument_name"] for inst in instruments
                   if inst["instrument_name"].startswith(f"BTC-{expiry_str}") and inst["instrument_name"].endswith(f"-{option_type}")])

def get_actual_iv(instrument_name):
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

def get_filtered_instruments(spot_price, expiry_str=DEFAULT_EXPIRY_STR, t_years=T_YEARS, multiplier=1):
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    if not calls_all:
        raise Exception(f"No call instruments found for expiry {expiry_str}.")
    strike_list = []
    for inst in calls_all:
        parts = inst.split("-")
        if len(parts) >= 3 and parts[2].isdigit():
            strike_list.append((inst, int(parts[2])))
    if not strike_list:
        raise Exception(f"No valid call instruments with strikes found for expiry {expiry_str}.")
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
    closest_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    nearest_instrument = strike_list[closest_index][0]
    actual_iv = get_actual_iv(nearest_instrument)
    if actual_iv is None:
        raise Exception("Could not fetch actual IV for the nearest instrument.")
    lower_bound = spot_price * np.exp(-actual_iv * np.sqrt(t_years) * multiplier)
    upper_bound = spot_price * np.exp(actual_iv * np.sqrt(t_years) * multiplier)
    filtered_calls = [inst for inst in calls_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_puts = [inst for inst in puts_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_calls.sort(key=lambda x: int(x.split("-")[2]))
    filtered_puts.sort(key=lambda x: int(x.split("-")[2]))
    if not filtered_calls and not filtered_puts:
        raise Exception("No instruments left after applying the theoretical range filter.")
    return filtered_calls, filtered_puts

###########################################
# Data Fetching Functions
###########################################
@st.cache_data(ttl=30)
def fetch_data(instruments_tuple):
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
    params = {"instrument_name": instrument_name}
    response = requests.get(URL_TICKER, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("result", {})

###########################################
# Kraken Data Fetch (Dual Timeframe)
###########################################
def fetch_kraken_data():
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

###########################################
# Volatility Calculations
###########################################
def calculate_ewma_roger_satchell_volatility(price_data, span=days_to_expiry):
    df = price_data.copy()
    df['rs'] = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
    ewma_rs = df['rs'].ewm(span=span, adjust=False).mean()
    return np.sqrt(ewma_rs.clip(lower=0))

def compute_realized_volatility_5min(df, annualize_days=365):
    df = df.copy()
    df['rs'] = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
    total_variance = df['rs'].sum()
    if total_variance <= 0:
        return 0.0
    N = len(df)
    M = annualize_days * 24 * 12
    annualization_factor = np.sqrt(M / N)
    return np.sqrt(total_variance) * annualization_factor

def calculate_btc_annualized_volatility_daily(df):
    if "date_time" not in df.columns:
        df = df.reset_index()
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    return daily_std * np.sqrt(365)

def calculate_daily_realized_volatility_series(df):
    if "date_time" not in df.columns:
        df = df.reset_index()
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    return df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)

###########################################
# Compute Daily Average IV and Historical VRP
###########################################
def compute_daily_average_iv(df_iv_agg):
    daily_iv = df_iv_agg["iv_mean"].resample("D").mean(numeric_only=True).dropna().tolist()
    return daily_iv

def compute_historical_vrp(daily_iv, daily_rv):
    n = min(len(daily_iv), len(daily_rv))
    return [(iv ** 2) - (rv ** 2) for iv, rv in zip(daily_iv[:n], daily_rv[:n])]

###########################################
# Option Delta and Gamma Calculation
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
    if sigma <= 0:
        return np.nan
    sigma_eff = sigma * risk_factor if 'risk_factor' in globals() else sigma
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
    if sigma <= 0:
        return np.nan
    sigma_eff = sigma * risk_factor if 'risk_factor' in globals() else sigma
    try:
        d1 = (np.log(S / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    except Exception:
        return np.nan
    gamma_val = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T))
    return gamma_val

def compute_gex(row, S, oi):
    gamma_val = compute_gamma(row, S)
    if gamma_val is None or np.isnan(gamma_val):
        return np.nan
    return gamma_val * oi * (S ** 2)

###########################################
# EV and Gamma Exposure (GEX) Calculations
###########################################
def compute_ev(adjusted_iv, rv, T, position_side):
    if position_side.lower() == "short":
        return (((adjusted_iv**2 - rv**2) * T) / 2) * 100
    else:
        return (((rv**2 - adjusted_iv**2) * T) / 2) * 100

def compute_gamma_value(spot, strike, sigma, T):
    d1 = (np.log(spot/strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (spot * sigma * np.sqrt(T))

def adjust_volatility_with_smile(strike, smile_df):
    sorted_smile = smile_df.sort_values("strike")
    strikes = sorted_smile["strike"].values
    ivs = sorted_smile["iv"].values
    return np.interp(strike, strikes, ivs)

def build_ticker_list_with_metrics(all_instruments, spot, T, smile_df, rv, position_side="short"):
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
        adjusted_iv = adjust_volatility_with_smile(strike, smile_df)
        try:
            d1 = (np.log(spot / strike) + 0.5 * adjusted_iv**2 * T) / (adjusted_iv * np.sqrt(T))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        ev_value = compute_ev(adjusted_iv, rv, T, position_side)  # Adaptive EV
        gamma_val = compute_gamma_value(spot, strike, adjusted_iv, T) if adjusted_iv > 0 and T > 0 else 0
        gex_value = gamma_val * ticker_data["open_interest"] * (spot**2)
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "delta": delta_est,
            "iv": adjusted_iv,
            "EV": ev_value,
            "gex": gex_value
        })
    return ticker_list

###########################################
# EV Strategy Functions
###########################################
def calculate_atm_straddle_ev(ticker_list, spot, T, rv, position_side="short"):
    tolerance = spot * 0.02  
    atm_candidates = [item for item in ticker_list if abs(item["strike"] - spot) <= tolerance]
    if not atm_candidates:
        return None
    atm_strikes = {}
    for item in atm_candidates:
        strike = item["strike"]
        if strike not in atm_strikes:
            atm_strikes[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            atm_strikes[strike]["iv_sum"] += item["iv"]
            atm_strikes[strike]["count"] += 1
    ev_candidates = []
    for strike, data in atm_strikes.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = compute_ev(avg_iv, rv, T, position_side)
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV", ascending=False)

def calculate_limited_otm_put_ev(ticker_list, spot, T, rv, position_side="long"):
    tolerance = spot * 0.10  
    candidates = [item for item in ticker_list if item["option_type"] == "P" and item["strike"] < spot and (spot - item["strike"]) <= tolerance]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = compute_ev(avg_iv, rv, T, position_side)
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV", ascending=False)

def calculate_call_spread_ev(ticker_list, spot, T, rv, position_side="short"):
    tolerance = spot * 0.10  
    candidates = [item for item in ticker_list if item["option_type"] == "C" and item["strike"] > spot and (item["strike"] - spot) <= tolerance]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = compute_ev(avg_iv, rv, T, position_side)
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV", ascending=False)

def calculate_strangle_ev(ticker_list, spot, T, rv, position_side="short"):
    tolerance = spot * 0.10  
    candidates = [item for item in ticker_list if ((item["option_type"] == "C" and item["strike"] > spot and (item["strike"] - spot) <= tolerance) or
                                                   (item["option_type"] == "P" and item["strike"] < spot and (spot - item["strike"]) <= tolerance))]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = compute_ev(avg_iv, rv, T, position_side)
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV", ascending=False)

def calculate_naked_call_ev(ticker_list, spot, T, rv, position_side="short"):
    candidates = [item for item in ticker_list if item["option_type"] == "C" and item["strike"] > spot]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = compute_ev(avg_iv, rv, T, position_side)
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV", ascending=False)

def calculate_small_atm_straddle_ev(ticker_list, spot, T, rv, position_side="short"):
    return calculate_atm_straddle_ev(ticker_list, spot, T, rv, position_side)

###########################################
# Composite Score Calculation
###########################################
def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

def compute_composite_scores(ticker_list, position_side='short'):
    for item in ticker_list:
        if 'gex' not in item:
            item['gex'] = 0
    ev_list = [item['EV'] for item in ticker_list]
    gex_list = [item.get('gex', 0) for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gex = normalize_metrics(gex_list)
    norm_oi = normalize_metrics(oi_list)
    weights = {"ev": 0.5, "gex": (-0.3 if position_side.lower() == 'short' else 0.3), "oi": 0.2}
    for i, item in enumerate(ticker_list):
        composite_score = (weights["ev"] * norm_ev[i] +
                           weights["gex"] * norm_gex[i] +
                           weights["oi"] * norm_oi[i])
        item["composite_score"] = composite_score
        item["strategy"] = position_side.capitalize()
    return ticker_list

###########################################
# Build Smile DataFrame from Ticker List
###########################################
def build_smile_df(ticker_list):
    df = pd.DataFrame(ticker_list)
    df = df.dropna(subset=["iv"])
    smile_df = df.groupby("strike", as_index=False)["iv"].mean()
    return smile_df

###########################################
# EV Update Function for Composite Table
###########################################
def update_ev_for_position(ticker_list, rv, T, position_side):
    for ticker in ticker_list:
        ticker["EV"] = compute_ev(ticker["iv"], rv, T, position_side)
    return ticker_list

###########################################
# Evaluate Trade Strategy
###########################################
def evaluate_trade_strategy(df, spot_price, risk_tolerance="Moderate", df_iv_agg_reset=None,
                            historical_vols=None, historical_vrps=None, expiry_date=None):
    current_date = dt.datetime.now()
    days_to_expiration = (expiry_date - current_date).days if expiry_date else 7
    rv_vol = calculate_btc_annualized_volatility_daily(df_kraken)
    iv_vol = df["iv_close"].mean() if not df.empty else np.nan
    rv_var = rv_vol ** 2 if not pd.isna(rv_vol) else np.nan
    iv_var = iv_vol ** 2 if not pd.isna(iv_vol) else np.nan
    current_vrp = iv_var - rv_var if (not pd.isna(iv_vol) and not pd.isna(rv_vol)) else np.nan
    divergence = abs(iv_vol - rv_vol) / rv_vol if (rv_vol != 0 and not pd.isna(iv_vol)) else np.nan
    if not np.isnan(divergence) and divergence > 0.20:
        st.write(f"Alert: IV and RV diverge by {divergence * 100:.2f}% (threshold: 20%).")
    df_iv_agg = df.groupby("date_time")["iv_close"].mean().to_frame(name="iv_mean")
    df_iv_agg.index = pd.to_datetime(df_iv_agg.index)
    df_iv_agg = df_iv_agg.resample("5min").mean().ffill().sort_index()
    df_iv_agg["iv_rolling_mean"] = df_iv_agg["iv_mean"].rolling("1D").mean()
    df_iv_agg["iv_rolling_std"] = df_iv_agg["iv_mean"].rolling("1D").std()
    df_iv_agg["upper_zone"] = df_iv_agg["iv_rolling_mean"] + df_iv_agg["iv_rolling_std"]
    df_iv_agg["lower_zone"] = df_iv_agg["iv_rolling_mean"] - df_iv_agg["iv_rolling_std"]
    latest_iv = df_iv_agg["iv_mean"].iloc[-1]
    latest_upper = df_iv_agg["upper_zone"].iloc[-1]
    latest_lower = df_iv_agg["lower_zone"].iloc[-1]
    if latest_iv > latest_upper:
        market_regime = "Risk-Off"
    elif latest_iv < latest_lower:
        market_regime = "Risk-On"
    else:
        market_regime = "Neutral"
    vol_regime = market_regime
    vrp_regime = classify_vrp_regime(current_vrp, historical_vols) if (historical_vols is not None and len(historical_vols) > 0) else "Neutral"
    call_items = [item for item in ticker_list if item["option_type"] == "C"]
    put_items = [item for item in ticker_list if item["option_type"] == "P"]
    call_oi_total = sum(item["open_interest"] for item in call_items)
    put_oi_total = sum(item["open_interest"] for item in put_items)
    avg_call_delta = (sum(item["delta"] * item["open_interest"] for item in call_items) / call_oi_total) if call_oi_total > 0 else 0
    avg_put_delta = (sum(item["delta"] * item["open_interest"] for item in put_items) / put_oi_total) if put_oi_total > 0 else 0
    df_ticker = pd.DataFrame(ticker_list) if ticker_list else pd.DataFrame()
    call_oi = df_ticker[df_ticker["option_type"] == "C"]["open_interest"].sum() if not df_ticker.empty else 0
    put_oi = df_ticker[df_ticker["option_type"] == "P"]["open_interest"].sum() if not df_ticker.empty else 0
    put_call_ratio = put_oi / call_oi if call_oi > 0 else np.inf
    df_calls = df[df["option_type"] == "C"].copy()
    df_puts = df[df["option_type"] == "P"].copy()
    if "gamma" not in df_calls.columns:
        df_calls["gamma"] = df_calls.apply(lambda row: compute_delta(row, spot_price), axis=1)
    if "gamma" not in df_puts.columns:
        df_puts["gamma"] = df_puts.apply(lambda row: compute_delta(row, spot_price), axis=1)
    avg_call_gamma = df_calls["gamma"].mean() if not df_calls.empty else 0
    avg_put_gamma = df_puts["gamma"].mean() if not df_puts.empty else 0
    if vrp_regime == "Long Volatility":
        if risk_tolerance == "Conservative":
            recommendation = "Long Volatility (Conservative): Buy limited OTM Puts"
            position = "Limited OTM Puts"
            hedge_action = "Hedge lightly with BTC futures short"
        elif risk_tolerance == "Moderate":
            recommendation = "Long Volatility (Neutral): Consider delta-hedged straddles"
            position = "ATM Straddle"
            hedge_action = "Implement dynamic delta hedging"
        else:
            recommendation = "Long Volatility (Aggressive): Take leveraged long volatility positions"
            position = "Leveraged Long Straddle"
            hedge_action = "Aggressively hedge using BTC futures"
    elif vrp_regime == "Short Volatility":
        if risk_tolerance == "Conservative":
            recommendation = "Short Volatility (Conservative): Sell a small number of call spreads"
            position = "Call Spread"
            hedge_action = "Hedge by buying BTC futures"
        elif risk_tolerance == "Moderate":
            recommendation = "Short Volatility (Neutral): Sell strangles with tight stops"
            position = "Strangle"
            hedge_action = "Monitor and adjust hedge dynamically"
        else:
            recommendation = "Short Volatility (Aggressive): Consider naked call selling"
            position = "Naked Calls"
            hedge_action = "Aggressively hedge with BTC futures"
    else:
        recommendation = "Gamma Scalping (Neutral): Consider buying small ATM straddles"
        position = "Small ATM Straddle"
        hedge_action = "Implement light delta hedging"
    return {
        "recommendation": recommendation,
        "position": position,
        "hedge_action": hedge_action,
        "iv": iv_vol,
        "rv": rv_vol,
        "vol_regime": vol_regime,
        "vrp_regime": vrp_regime,
        "put_call_ratio": put_call_ratio,
        "avg_call_delta": avg_call_delta,
        "avg_put_delta": avg_put_delta,
        "avg_call_gamma": avg_call_gamma,
        "avg_put_gamma": avg_put_gamma
    }

def classify_vrp_regime(current_vrp, historical_vrps):
    percentile = percentileofscore(historical_vrps, current_vrp)
    if current_vrp < 0:
        return "Long Volatility"
    elif percentile > 75:
        return "Short Volatility"
    elif percentile < 25:
        return "Long Volatility"
    else:
        return "Neutral"

###########################################
# Visualization Functions
###########################################
def plot_gamma_heatmap(df):
    st.subheader("Gamma Heatmap by Strike and Time")
    if "gamma" not in df.columns:
        st.error("DataFrame missing 'gamma' column for heatmap.")
        return
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

def plot_net_gex(df_gex, spot_price):
    st.subheader("Net Gamma Exposure by Strike")
    df_gex_net = df_gex.groupby("strike").apply(
        lambda x: x.loc[x["option_type"] == "C", "gex"].sum() - x.loc[x["option_type"] == "P", "gex"].sum()
    ).reset_index(name="net_gex")
    df_gex_net["sign"] = df_gex_net["net_gex"].apply(lambda val: "Negative" if val < 0 else "Positive")
    fig_net_gex = px.bar(
        df_gex_net,
        x="strike",
        y="net_gex",
        color="sign",
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
    st.plotly_chart(fig_net_gex, use_container_width=True)

def classify_volatility_regime(current_vol, historical_vols):
    percentile = percentileofscore(historical_vols, current_vol)
    if percentile < 5:
        return "Low Volatility"
    elif percentile > 95:
        return "High Volatility"
    else:
        return "Medium Volatility"

def plot_delta_balance(ticker_list, spot_price):
    st.subheader("Put vs Call Delta Balance")
    
    if not ticker_list:
        st.warning("No ticker data available to calculate delta balance.")
        return
        
    call_items = [item for item in ticker_list if item["option_type"] == "C"]
    put_items = [item for item in ticker_list if item["option_type"] == "P"]
    
    call_weighted_delta = sum(item["delta"] * item["open_interest"] for item in call_items)
    put_weighted_delta = abs(sum(item["delta"] * item["open_interest"] for item in put_items))
    
    delta_data = pd.DataFrame({
        'Option Type': ['Calls', 'Puts'],
        'Total Weighted Delta': [call_weighted_delta, put_weighted_delta],
        'Direction': ['Bullish', 'Bearish']
    })
    
    fig = px.bar(
        delta_data,
        x='Option Type',
        y='Total Weighted Delta',
        color='Direction',
        color_discrete_map={'Bullish': 'green', 'Bearish': 'red'},
        title='Put vs Call Delta Balance (Open Interest Weighted)',
        labels={'Total Weighted Delta': 'Absolute Total Delta * Open Interest'}
    )
    
    net_delta = call_weighted_delta - put_weighted_delta
    bias_strength = abs(net_delta)
    if net_delta > 0:
        bias_text = f"Market Bias: Bullish (Net Delta: +{bias_strength:.2f})"
        bias_color = "green"
    else:
        bias_text = f"Market Bias: Bearish (Net Delta: {net_delta:.2f})"
        bias_color = "red"
    
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text=bias_text,
        showarrow=False,
        font=dict(size=14, color=bias_color),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=bias_color,
        borderwidth=2,
        borderpad=4
    )
    
    if put_weighted_delta > 0:
        delta_ratio = call_weighted_delta / put_weighted_delta
        fig.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Call/Put Delta Ratio: {delta_ratio:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="center",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    
    fig_strikes = create_delta_by_strike_chart(call_items, put_items, spot_price)
    
    st.plotly_chart(fig, use_container_width=True)
    if fig_strikes:
        st.plotly_chart(fig_strikes, use_container_width=True)

def create_delta_by_strike_chart(call_items, put_items, spot_price):
    ranges = [
        {"name": "Deep ITM", "min_pct": -float('inf'), "max_pct": -0.10},
        {"name": "ITM", "min_pct": -0.10, "max_pct": -0.02},
        {"name": "Near ATM", "min_pct": -0.02, "max_pct": 0.02},
        {"name": "OTM", "min_pct": 0.02, "max_pct": 0.10},
        {"name": "Deep OTM", "min_pct": 0.10, "max_pct": float('inf')}
    ]
    
    range_data = []
    for strike_range in ranges:
        min_strike = spot_price * (1 + strike_range["min_pct"])
        max_strike = spot_price * (1 + strike_range["max_pct"])
        calls_in_range = [item for item in call_items if min_strike <= item["strike"] < max_strike]
        call_delta = sum(item["delta"] * item["open_interest"] for item in calls_in_range)
        range_data.append({
            "Strike Range": strike_range["name"],
            "Option Type": "Calls",
            "Weighted Delta": call_delta,
            "Sort Order": ranges.index(strike_range)
        })
    for strike_range in ranges:
        min_strike = spot_price * (1 + strike_range["min_pct"])
        max_strike = spot_price * (1 + strike_range["max_pct"])
        puts_in_range = [item for item in put_items if min_strike <= item["strike"] < max_strike]
        put_delta = abs(sum(item["delta"] * item["open_interest"] for item in puts_in_range))
        range_data.append({
            "Strike Range": strike_range["name"],
            "Option Type": "Puts",
            "Weighted Delta": put_delta,
            "Sort Order": ranges.index(strike_range)
        })
    
    df_range = pd.DataFrame(range_data)
    df_range = df_range.sort_values("Sort Order")
    fig = px.bar(
        df_range,
        x="Strike Range",
        y="Weighted Delta",
        color="Option Type",
        barmode="group",
        title="Delta Distribution by Strike Range",
        color_discrete_map={"Calls": "green", "Puts": "red"},
        category_orders={"Strike Range": [r["name"] for r in ranges]}
    )
    return fig

###########################################
# MAIN DASHBOARD
###########################################
def main():
    login()
    st.title("Crypto Options Visualization Dashboard with Enhanced Risk Adjustments")
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
    T_YEARS = days_to_expiry / 365
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
    
    global risk_factor
    risk_factor = compute_risk_adjustment_factor_cf(df_kraken, alpha=0.05)
    st.write(f"Risk Adjustment Factor (CF): {risk_factor:.2f}")
    
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
        preliminary_ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "iv": raw_iv
        })
    smile_df = build_smile_df(preliminary_ticker_list)
    
    # Calculate realized volatility (rv) from Kraken data
    rv = calculate_btc_annualized_volatility_daily(df_kraken)
    
    global ticker_list
    ticker_list = build_ticker_list_with_metrics(all_instruments, spot_price, T_YEARS, smile_df, rv, position_side="short")
    
    # Create a DataFrame from ticker_list for hedge calculations
    df_ticker = pd.DataFrame(ticker_list)
    
    daily_rv_series = calculate_daily_realized_volatility_series(df_kraken)
    daily_rv = daily_rv_series.tolist()
    daily_iv = compute_daily_average_iv(df_iv_agg)
    historical_vrps = compute_historical_vrp(daily_iv, daily_rv)
    
    st.subheader("Volatility Trading Decision Tool")
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance",
                                          options=["Conservative", "Moderate", "Aggressive"],
                                          index=1)
    
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
    
    # Futures Hedge Recommendation based on net delta
    net_delta = df_ticker.assign(weighted_delta=df_ticker["delta"] * df_ticker["open_interest"])["weighted_delta"].sum()
    if net_delta > 0:
        futures_hedge = "Short BTC Futures"
    elif net_delta < 0:
        futures_hedge = "Long BTC Futures"
    else:
        futures_hedge = "No hedge required"
    hedge_table = pd.DataFrame({
        "Metric": ["Net Delta", "Futures Hedge Recommendation"],
        "Value": [net_delta, futures_hedge]
    })
    st.subheader("Futures Hedge Recommendation")
    st.dataframe(hedge_table.style.hide(axis="index"))
    
    rv_series = calculate_daily_realized_volatility_series(df_kraken)
    rv_scalar = rv_series.iloc[-1] if not rv_series.empty else np.nan
    position = trade_decision['position']
    if position in ["ATM Straddle", "Leveraged Long Straddle"]:
        df_ev = calculate_atm_straddle_ev(ticker_list, spot_price, T_YEARS, rv_scalar, position_side="long")
        st.subheader("ATM Straddle EV Analysis")
    elif position == "Limited OTM Puts":
        df_ev = calculate_limited_otm_put_ev(ticker_list, spot_price, T_YEARS, rv_scalar, position_side="long")
        st.subheader("Limited OTM Put EV Analysis")
    elif position == "Call Spread":
        df_ev = calculate_call_spread_ev(ticker_list, spot_price, T_YEARS, rv_scalar, position_side="short")
        st.subheader("Call Spread EV Analysis")
    elif position == "Strangle":
        df_ev = calculate_strangle_ev(ticker_list, spot_price, T_YEARS, rv_scalar, position_side="short")
        st.subheader("Strangle EV Analysis")
    elif position == "Naked Calls":
        df_ev = calculate_naked_call_ev(ticker_list, spot_price, T_YEARS, rv_scalar, position_side="short")
        st.subheader("Naked Call EV Analysis")
    elif position == "Small ATM Straddle":
        df_ev = calculate_small_atm_straddle_ev(ticker_list, spot_price, T_YEARS, rv_scalar, position_side="short")
        st.subheader("Small ATM Straddle EV Analysis")
    else:
        df_ev = None
        st.subheader("EV Analysis")
        st.write("EV analysis for the selected position is not implemented yet.")
    
    # EV Analysis Block
    if df_ev is not None and not df_ev.empty and not df_ev["EV"].isna().all():
        df_ev_clean = df_ev.dropna(subset=["EV"])
        if not df_ev_clean.empty:
            best_candidate = df_ev_clean.loc[df_ev_clean["EV"].idxmax()]
            best_strike = best_candidate["Strike"]
            st.write("Candidate Strikes and their Expected Value (EV):")
            st.dataframe(df_ev_clean.style.hide(axis="index"))
            st.write(f"Recommended Strike based on highest EV: {best_strike}")
        else:
            st.write("No candidates found with valid EV values.")
    else:
        st.write("No candidates found within tolerance for EV calculation.")
    
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
        plot_delta_balance(ticker_list, spot_price)
        if "gamma" not in df_calls.columns:
            df_calls["gamma"] = df_calls.apply(lambda row: compute_delta(row, spot_price), axis=1)
        if "gamma" not in df_puts.columns:
            df_puts["gamma"] = df_puts.apply(lambda row: compute_delta(row, spot_price), axis=1)
        plot_gamma_heatmap(pd.concat([df_calls, df_puts]))
    
    # Only display the Net GEX chart (no separate GEX table)
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
        plot_net_gex(df_gex, spot_price)
    
    ###########################################
    # Composite Score Table (Using Actual Data)
    ###########################################
    short_ticker_list = update_ev_for_position(ticker_list.copy(), rv, T_YEARS, "short")
    long_ticker_list = update_ev_for_position(ticker_list.copy(), rv, T_YEARS, "long")
    short_scores = compute_composite_scores(short_ticker_list, position_side="short")
    long_scores = compute_composite_scores(long_ticker_list, position_side="long")
    df_short = pd.DataFrame(short_scores)
    df_long = pd.DataFrame(long_scores)
    df_combined = pd.concat([df_short, df_long], ignore_index=True)
    if "EV" in df_combined.columns and "open_interest" in df_combined.columns:
        columns_to_show = ["instrument", "strike", "option_type", "open_interest", "delta", "iv", "EV", "gex", "composite_score", "strategy"]
        df_combined = df_combined[columns_to_show]
        st.subheader("Combined Composite Score Table")
        st.dataframe(df_combined.style.hide(axis="index"))
    else:
        st.write("Composite score data is not available.")

if __name__ == '__main__':
    main()
