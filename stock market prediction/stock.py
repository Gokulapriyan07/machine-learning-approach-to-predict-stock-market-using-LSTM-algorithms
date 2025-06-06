import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
import time
from datetime import datetime, timedelta

# Fetch S&P 500 tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]["Symbol"].tolist()

# Exchange rates API
@st.cache_data(ttl=600)
def fetch_exchange_rates(base="USD"):
    try:
        response = requests.get(f"https://api.exchangerate.host/latest?base={base}")
        return response.json().get("rates", {})
    except:
        return {}

@st.cache_data(ttl=600)
def convert_currency(amount, from_curr, to_curr):
    rates = fetch_exchange_rates(from_curr)
    return amount * rates.get(to_curr, 1)

# Cache ticker object
@st.cache_resource(ttl=600)
def load_ticker(ticker):
    return yf.Ticker(ticker)

# Cache historical data
@st.cache_data(ttl=600)
def get_ticker_data(ticker, period):
    return load_ticker(ticker).history(period=period)

# Live price fetcher
@st.cache_data(ttl=600)
def get_live_price(ticker):
    for _ in range(5):
        try:
            data = load_ticker(ticker).history(period='1d')
            if data.empty:
                raise ValueError("No data returned.")
            return data['Close'].values[-1]
        except Exception as e:
            if "Too Many Requests" in str(e):
                st.warning("Rate limit hit. Retrying...")
                time.sleep(5)
            else:
                st.warning(f"Live price not available: {e}")
                return None
    st.warning("Rate limit exceeded too many times.")
    return None

# Get future trading dates
def get_future_trading_dates(n):
    dates, current = [], datetime.now()
    while len(dates) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
    return dates

# ---------------------- UI ---------------------- #
st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.sidebar.title("Stock Market Dashboard")

symbols = get_sp500_tickers()
stock = st.sidebar.selectbox("Select Stock", symbols)
currency = st.sidebar.selectbox("Currency", ["INR"])
currency_symbol = {"INR": "₹"}.get(currency, "")
base_currency = "USD"

live_price = get_live_price(stock)
if live_price:
    converted_price = convert_currency(live_price, base_currency, currency)
    st.sidebar.markdown(f"**Live Price:** <span style='font-size:20px;'>{currency_symbol}{converted_price:.2f}</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("**Live Price:** Not available")

range_options = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y"}
date_range = st.sidebar.selectbox("Date Range", list(range_options.keys()))
prediction_days = st.sidebar.slider("Prediction Days", 10, 90, 30)

tabs = st.tabs(["Stock Predictor", "Stock Summary"])
ticker_data = load_ticker(stock)

# ------------------ Predictor Tab ------------------ #
with tabs[0]:
    st.title("📈 Stock Market Predictor")
    try:
        data = get_ticker_data(stock, range_options[date_range])
        df = data[['Close']]
        if len(df) < 90:
            data = get_ticker_data(stock, "6mo")
            df = data[['Close']]

        st.subheader("Price History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[convert_currency(v, base_currency, currency) for v in data['Close']],
            mode='lines',
            name='Close',
            line=dict(width=3, color='blue')))
        fig.update_layout(
            xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})",
            template="plotly_white", height=500)
        st.plotly_chart(fig)

        if st.button("Predict"):
            st.write("Training model...")

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df)

            x_train, y_train = [], []
            for i in range(60, len(scaled)):
                x_train.append(scaled[i-60:i, 0])
                y_train.append(scaled[i, 0])

            if not x_train:
                st.error("Insufficient data to train model.")
            else:
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

                model = Sequential([
                    LSTM(100, return_sequences=True, input_shape=(60, 1)),
                    Dropout(0.2),
                    LSTM(100),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=0)

                input_seq = scaled[-60:]
                preds = []
                for _ in range(prediction_days):
                    pred = model.predict(input_seq.reshape(1, 60, 1), verbose=0)[0][0]
                    preds.append(pred)
                    input_seq = np.append(input_seq[1:], [[pred]], axis=0)

                inv_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                inv_preds = [convert_currency(p, base_currency, currency) for p in inv_preds]
                future_dates = get_future_trading_dates(prediction_days)

                st.subheader("Predicted Prices")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=future_dates, y=inv_preds, mode='lines', line=dict(color='red', width=3)))
                fig_pred.update_layout(xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})", template="plotly_white")
                st.plotly_chart(fig_pred)

                pred_table = pd.DataFrame({
                    "Date": future_dates,
                    f"Predicted Price ({currency})": [f"{currency_symbol}{p:.2f}" for p in inv_preds]
                })
                st.dataframe(pred_table, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

# ------------------ Summary Tab ------------------ #
with tabs[1]:
    st.title("📊 Stock Summary")
    try:
        info = ticker_data.info
        col1, col2, col3 = st.columns(3)
        col1.metric("Open", f"{currency_symbol}{convert_currency(info.get('open', 0), base_currency, currency):.2f}")
        col2.metric("Close", f"{currency_symbol}{convert_currency(info.get('previousClose', 0), base_currency, currency):.2f}")
        col3.metric("High", f"{currency_symbol}{convert_currency(info.get('dayHigh', 0), base_currency, currency):.2f}")
        col1.metric("Low", f"{currency_symbol}{convert_currency(info.get('dayLow', 0), base_currency, currency):.2f}")
        col2.metric("Market Cap", f"{currency_symbol}{convert_currency(info.get('marketCap', 0), base_currency, currency):,.2f}")
        col3.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
    except Exception as e:
        st.error(f"Failed to load summary: {e}")
