import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import pandas as pd
import numpy as np
import plotly.express as px


# ---------- Page Config ----------
st.set_page_config(
    page_title="Stock Time-Series Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Market Time-Series Dashboard")
st.markdown(
    "Data source: **Yahoo Finance** via `yfinance` (when available). "
    "If the data provider rate-limits us, we fall back to a synthetic demo dataset."
)


# ---------- Sidebar Controls ----------
st.sidebar.header("Settings")

default_ticker = "AAPL"
ticker = st.sidebar.text_input(
    "Stock Ticker (e.g., AAPL, MSFT, TSLA, GOOGL)",
    default_ticker
).upper()

period = st.sidebar.selectbox(
    "Select Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=3,  # default "1y"
)

interval = st.sidebar.selectbox(
    "Select Interval",
    ["1d", "1wk", "1mo"],
    index=0,  # default "1d"
)

indicators = st.sidebar.multiselect(
    "Show Indicators",
    ["Moving Averages", "Bollinger Bands", "Returns", "RSI"],
    default=["Moving Averages", "Returns"]
)


# ---------- Helper Functions ----------
@st.cache_data
def load_data_from_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker using yfinance.
    Uses Ticker().history to avoid MultiIndex column issues.
    May raise YFRateLimitError or other exceptions.
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    df = df.reset_index()  # Date becomes a column
    return df


@st.cache_data
def generate_synthetic_data(ticker: str, n_points: int = 252) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV time-series that looks like a stock price.
    This is used as a fallback when yfinance is rate-limited.
    """
    np.random.seed(42)  # so demo is reproducible
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_points, freq="B")

    # Simulate log-returns and build a price path
    mu = 0.0005   # small drift
    sigma = 0.02  # daily volatility
    log_returns = np.random.normal(mu, sigma, size=n_points)
    price = 100 * np.exp(np.cumsum(log_returns))  # start at 100

    # Build OHLC around the "Close" price
    close = price
    open_ = close * (1 + np.random.normal(0, 0.002, size=n_points))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.005, size=n_points)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.005, size=n_points)))
    volume = np.random.randint(1e5, 5e5, size=n_points)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    df["Ticker"] = ticker
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) for a price series.
    """
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain).rolling(window).mean()
    roll_down = pd.Series(loss).rolling(window).mean()

    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=series.index)


# ---------- Main Data Loading ----------
if ticker.strip() == "":
    st.warning("Please enter a valid ticker symbol in the sidebar.")
    st.stop()

data_source = "yfinance"

try:
    df = load_data_from_yfinance(ticker, period, interval)
except YFRateLimitError:
    st.warning(
        "⚠️ Yahoo Finance rate limit reached for this app. "
        "Switching to a synthetic demo dataset so the dashboard still works."
    )
    df = generate_synthetic_data(ticker)
    data_source = "synthetic"
except Exception as e:
    st.warning(
        f"⚠️ Could not load data from Yahoo Finance due to: {type(e).__name__}. "
        "Using a synthetic demo dataset instead."
    )
    df = generate_synthetic_data(ticker)
    data_source = "synthetic"

if df.empty:
    st.error("No data found or generated. Please try another ticker or refresh.")
    st.stop()

# Ensure expected columns exist
expected_cols = {"Open", "High", "Low", "Close", "Volume"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"Data does not contain expected columns: {expected_cols}")
    st.dataframe(df.head())
    st.stop()

st.caption(f"Current data source: **{data_source}**")


# ---------- Feature Engineering ----------
df = df.sort_values("Date").reset_index(drop=True)

df["Return_pct"] = df["Close"].pct_change() * 100.0
df["MA_20"] = df["Close"].rolling(window=20).mean()
df["MA_50"] = df["Close"].rolling(window=50).mean()

# Bollinger Bands (20-period)
rolling_std = df["Close"].rolling(window=20).std()
df["BB_Middle"] = df["MA_20"]
df["BB_Upper"] = df["MA_20"] + (2 * rolling_std)
df["BB_Lower"] = df["MA_20"] - (2 * rolling_std)

# RSI
df["RSI_14"] = compute_rsi(df["Close"], window=14)

# Drop initial rows with NaNs (from rolling windows)
df_clean = df.dropna().reset_index(drop=True)


# ---------- Top KPIs ----------
latest_row = df_clean.iloc[-1]

latest_close = latest_row["Close"]
max_close = df_clean["Close"].max()
min_close = df_clean["Close"].min()
avg_volume = df_clean["Volume"].mean()

# Simple volatility measure from returns
ann_vol = df_clean["Return_pct"].std()

st.subheader(f"Summary for **{ticker}** ({period}, {interval})")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Latest Close", f"{latest_close:,.2f}")
kpi2.metric("Max Close (Period)", f"{max_close:,.2f}")
kpi3.metric("Min Close (Period)", f"{min_close:,.2f}")
kpi4.metric("Volatility (Std of returns)", f"{ann_vol:,.2f} %")


# ---------- Charts ----------
price_tab, volume_tab, returns_tab, rsi_tab, data_tab = st.tabs(
    ["📊 Price", "📦 Volume", "📉 Returns", "📐 RSI", "📄 Data"]
)

# --- Price Tab ---
with price_tab:
    st.markdown("### Price Chart")

    y_cols = ["Close"]
    if "Moving Averages" in indicators:
        y_cols.extend(["MA_20", "MA_50"])
    if "Bollinger Bands" in indicators:
        y_cols.extend(["BB_Upper", "BB_Lower"])

    fig_price = px.line(
        df_clean,
        x="Date",
        y=y_cols,
        labels={"value": "Price", "Date": "Date", "variable": "Series"},
        title=f"{ticker} - Price & Indicators",
    )
    st.plotly_chart(fig_price, use_container_width=True)


# --- Volume Tab ---
with volume_tab:
    st.markdown("### Trading Volume")
    fig_vol = px.bar(
        df_clean,
        x="Date",
        y="Volume",
        labels={"Volume": "Volume", "Date": "Date"},
        title=f"{ticker} - Trading Volume",
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# --- Returns Tab ---
with returns_tab:
    st.markdown("### Daily Returns (%)")
    if "Returns" in indicators:
        fig_ret = px.line(
            df_clean,
            x="Date",
            y="Return_pct",
            labels={"Return_pct": "Return (%)", "Date": "Date"},
            title=f"{ticker} - Daily Returns",
        )
        st.plotly_chart(fig_ret, use_container_width=True)
    else:
        st.info("Enable 'Returns' indicator in the sidebar to view this chart.")


# --- RSI Tab ---
with rsi_tab:
    st.markdown("### RSI (14-period)")
    if "RSI" in indicators:
        fig_rsi = px.line(
            df_clean,
            x="Date",
            y="RSI_14",
            labels={"RSI_14": "RSI", "Date": "Date"},
            title=f"{ticker} - RSI (14)",
        )
        fig_rsi.add_hrect(y0=30, y1=70, line_width=0, fillcolor="lightgray", opacity=0.2)
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.caption("RSI above 70 ≈ overbought, below 30 ≈ oversold (rule of thumb).")
    else:
        st.info("Enable 'RSI' indicator in the sidebar to view this chart.")


# --- Data Tab ---
with data_tab:
    st.markdown("### Raw & Engineered Data")
    st.dataframe(df_clean.tail(50))

    # Download button
    csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download data as CSV",
        data=csv,
        file_name=f"{ticker}_time_series_{data_source}.csv",
        mime="text/csv",
    )
