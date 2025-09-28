import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from statsmodels.tsa.arima.model import ARIMA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from firebase_admin import credentials, initialize_app, firestore, auth
import firebase_admin.exceptions
import os

# --- API KEY & CONFIGURATION ---
# Replace this with your actual Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"

# Firebase configuration
# Replace this with your actual Firebase config JSON object
FIREBASE_CONFIG = {
    "apiKey": "YOUR_FIREBASE_API_KEY",
    "authDomain": "YOUR_FIREBASE_AUTH_DOMAIN",
    "projectId": "YOUR_FIREBASE_PROJECT_ID",
    "storageBucket": "YOUR_FIREBASE_STORAGE_BUCKET",
    "messagingSenderId": "YOUR_FIREBASE_MESSAGING_SENDER_ID",
    "appId": "YOUR_FIREBASE_APP_ID"
}

# --- INITIALIZE FIREBASE & SESSION STATE ---
if 'firebase_app_initialized' not in st.session_state:
    try:
        # NOTE: For deployment, 'firebase-key.json' must be available on the server (e.g., via Streamlit Secrets).
        cred = credentials.Certificate("firebase-key.json")
        initialize_app(cred)
        st.session_state['db'] = firestore.client()
        st.session_state['firebase_app_initialized'] = True
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.stop()

if 'auth_status' not in st.session_state:
    st.session_state['auth_status'] = None
    st.session_state['user_id'] = None

# --- DATA CLEANING FUNCTION ---
def safe_float(value, default=0.0):
    """Safely converts a string to a float, handling 'None', 'NaN', and other non-numeric strings."""
    if isinstance(value, (int, float)):
        return value
    if value in ['None', 'NaN', 'N/A', None, '']:
        return default
    try:
        return float(value)
    except:
        return default

# --- DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def fetch_current_price(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Global Quote' in data and data['Global Quote']:
            return safe_float(data['Global Quote']['05. price'])
        else:
            st.warning(f"No current price data found for {symbol}.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price for {symbol}: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_historical_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (Daily)' in data:
            # Use '4. close' for general price data, converting strings to float safely
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df['Close'] = df['4. close'].apply(safe_float)
            df.index = pd.to_datetime(df.index)
            return df[['Close']].replace(0, np.nan).dropna() # Remove zero values that might skew analysis
        else:
            st.warning(f"No historical data found for {symbol}.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_news_sentiment(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'feed' in data:
            return data['feed']
        return []
    except Exception as e:
        st.error(f"Error fetching news sentiment for {symbol}: {e}")
        return []

# --- AUTHENTICATION FUNCTIONS ---
def login_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        # Simplified login for demo purposes. In a production app, use
        # a secure Firebase method to verify the password.
        st.session_state['auth_status'] = 'authenticated'
        st.session_state['user_id'] = user.uid
        st.rerun()
    except firebase_admin.exceptions.NotFoundError:
        st.error("Invalid email or password.")
    except Exception as e:
        st.error(f"Login failed: {e}")

def logout_user():
    st.session_state['auth_status'] = None
    st.session_state['user_id'] = None
    st.rerun()

# --- PORTFOLIO DATA MANAGEMENT ---
def load_portfolio(user_id):
    if 'db' in st.session_state:
        doc_ref = st.session_state['db'].collection('users').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get('portfolio', {})
    return {}

def save_portfolio(user_id, portfolio):
    if 'db' in st.session_state:
        doc_ref = st.session_state['db'].collection('users').document(user_id)
        doc_ref.set({'portfolio': portfolio})
        st.success("Portfolio saved successfully!")

# --- UI FOR AUTHENTICATION ---
if 'firebase_app_initialized' not in st.session_state:
    st.title("Initializing Application...")
    st.stop()
elif st.session_state['auth_status'] != 'authenticated':
    st.sidebar.title("Login")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        login_user(email, password)
    
    st.title("Please log in to continue")
    st.stop()
else:
    st.sidebar.title("Dashboard")
    st.sidebar.write(f"Welcome, User {st.session_state['user_id'][:5]}...")
    if st.sidebar.button("Logout"):
        logout_user()

    # --- THE REST OF YOUR APPLICATION CODE ---
    
    # Portfolio data, now loaded from Firestore
    portfolio = load_portfolio(st.session_state['user_id'])
    
    # --- PORTFOLIO MANAGEMENT SECTION ---
    st.header("Portfolio Management")
    new_ticker = st.text_input("Add new stock ticker (e.g., TSLA)")
    new_shares = st.number_input("Number of shares", min_value=1, step=1)
    
    col_add, col_save = st.columns(2)
    with col_add:
        if st.button("Add Stock"):
            if new_ticker and new_shares:
                portfolio[new_ticker.upper()] = {'shares': new_shares}
                save_portfolio(st.session_state['user_id'], portfolio)
                st.success(f"{new_ticker.upper()} with {new_shares} shares added to your portfolio.")
                st.rerun()
            else:
                st.error("Please enter both a ticker and the number of shares.")

    with col_save:
        if st.button("Save Portfolio to Cloud"):
            save_portfolio(st.session_state['user_id'], portfolio)

    if not portfolio:
        st.info("Your portfolio is empty. Add some stocks to get started.")
        st.stop()

    # Fetch real-time data for all tickers in the portfolio
    tickers = list(portfolio.keys())
    current_prices = {}
    prev_prices = {}
    all_hist_data = pd.DataFrame()

    for ticker in tickers:
        price = fetch_current_price(ticker, ALPHA_VANTAGE_API_KEY)
        if price is not None:
            current_prices[ticker] = price

        hist_df = fetch_historical_data(ticker, ALPHA_VANTAGE_API_KEY)
        if not hist_df.empty:
            all_hist_data[ticker] = hist_df['Close']
    
    if not current_prices:
        st.error("Could not fetch any data. Please check your API key and connection.")
        st.stop()

    # Build the portfolio DataFrame
    portfolio_df = pd.DataFrame.from_dict(portfolio, orient='index')
    portfolio_df.index.name = 'Ticker'
    portfolio_df['Current Price'] = pd.Series(current_prices)
    portfolio_df['Current Value'] = portfolio_df['Current Price'] * portfolio_df['shares']

    # --- PORTFOLIO OVERVIEW SECTION ---
    st.header("Portfolio Overview")
    
    total_value = portfolio_df['Current Value'].sum()
    prev_value = (pd.Series(prev_prices) * portfolio_df['shares']).sum()
    daily_change = ((total_value - prev_value) / prev_value) * 100 if prev_value != 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}", f"{daily_change:,.2f}%")

    with col2:
        # Asset Allocation Pie Chart
        fig_pie = px.pie(values=portfolio_df['Current Value'], names=portfolio_df.index, title='Asset Allocation')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Holdings Table
    st.subheader("Holdings")
    st.dataframe(portfolio_df[['shares', 'Current Price', 'Current Value']].style.format({"Current Price": "${:,.2f}", "Current Value": "${:,.2f}"}))
    
    # --- RISK METRICS SECTION ---
    st.header("Risk Metrics")
    if not all_hist_data.empty:
        # Calculate daily returns
        returns = all_hist_data.pct_change()

        # Calculate portfolio returns
        weights = np.array(portfolio_df['Current Value'] / total_value)
        portfolio_returns = returns.dot(weights)

        # Calculate annualized volatility
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Assume a risk-free rate (e.g., 10-year treasury yield)
        risk_free_rate = 0.04 # 4%
        
        # Calculate annualized Sharpe Ratio
        sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / volatility

        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            st.metric("Portfolio Volatility (Annualized)", f"{volatility:.2%}")
        with col_risk2:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # --- INDIVIDUAL STOCK ANALYSIS SECTION ---
    st.header("Individual Stock Analysis")
    selected_ticker = st.selectbox("Select a stock to analyze", tickers)

    if selected_ticker:
        hist_data = all_hist_data[[selected_ticker]].dropna().rename(columns={selected_ticker: 'Close'})

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Historical Performance")

            # Technical Indicators UI
            show_sma_20 = st.checkbox('Show 20-Day SMA')
            show_sma_50 = st.checkbox('Show 50-Day SMA')
            show_forecast = st.checkbox('Show 30-Day Forecast')
            
            # Create the Plotly figure
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], mode='lines', name='Price'))
            
            # Add technical indicators if selected
            if show_sma_20:
                hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
                fig_hist.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA_20'], mode='lines', name='20-Day SMA'))
                
            if show_sma_50:
                hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
                fig_hist.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA_50'], mode='lines', name='50-Day SMA'))

            # Add predictive analytics if selected
            if show_forecast:
                try:
                    # Use a simple ARIMA model
                    model = ARIMA(hist_data['Close'], order=(5, 1, 0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=30)
                    
                    forecast_dates = pd.date_range(start=hist_data.index[-1], periods=31, freq='D')[1:]
                    fig_hist.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='30-Day Forecast', line=dict(dash='dash')))
                except Exception as e:
                    st.warning(f"Could not generate forecast: {e}")


            fig_hist.update_layout(title=f"{selected_ticker} Price (1 Year)", xaxis_title="Date", yaxis_title="Price ($)")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col4:
            st.subheader("Key Fundamentals")
            # Fetch fundamental data from a different Alpha Vantage endpoint
            fundamental_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={selected_ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
            try:
                fundamental_response = requests.get(fundamental_url)
                fundamental_response.raise_for_status()
                fundamental_info = fundamental_response.json()
                
                # Use safe_float for all fundamental values
                market_cap = safe_float(fundamental_info.get('MarketCapitalization', 0))
                pe_ratio = safe_float(fundamental_info.get('PERatio', 0))
                dividend_yield = safe_float(fundamental_info.get('DividendYield', 0)) * 100
                week_high = safe_float(fundamental_info.get('52WeekHigh', 0))

                st.write(f"**Market Cap:** ${market_cap:,.0f}")
                st.write(f"**P/E Ratio:** {pe_ratio:.2f}")
                st.write(f"**Dividend Yield:** {dividend_yield:.2f}%")
                st.write(f"**52-Week High:** ${week_high:.2f}")
            except Exception as e:
                st.error(f"Error fetching fundamental data: {e}")

        # --- NEWS & SENTIMENT ANALYSIS SECTION ---
        st.header("News & Sentiment Analysis")
        news_feed = fetch_news_sentiment(selected_ticker, ALPHA_VANTAGE_API_KEY)
        
        if news_feed:
            analyzer = SentimentIntensityAnalyzer()
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            st.subheader("Latest Headlines")
            for article in news_feed[:5]:
                headline = article.get('title', 'No title')
                summary = article.get('summary', 'No summary')
                sentiment_score = analyzer.polarity_scores(headline)
                
                if sentiment_score['compound'] >= 0.05:
                    positive_count += 1
                    sentiment_label = "Positive"
                    st.markdown(f"**<span style='color:green'>▲ {headline}</span>**", unsafe_allow_html=True)
                elif sentiment_score['compound'] <= -0.05:
                    negative_count += 1
                    sentiment_label = "Negative"
                    st.markdown(f"**<span style='color:red'>▼ {headline}</span>**", unsafe_allow_html=True)
                else:
                    neutral_count += 1
                    sentiment_label = "Neutral"
                    st.markdown(f"**<span style='color:orange'>• {headline}</span>**", unsafe_allow_html=True)
                
                with st.expander(f"Read more about this {sentiment_label} headline"):
                    st.write(f"Source: [{article.get('source', 'N/A')}]({article.get('url', '#')})")
                    st.write(summary)
            
            st.subheader("Sentiment Summary")
            col_sent1, col_sent2, col_sent3 = st.columns(3)
            with col_sent1:
                st.metric("Positive", positive_count)
            with col_sent2:
                st.metric("Negative", negative_count)
            with col_sent3:
                st.metric("Neutral", neutral_count)
        else:
            st.warning("Could not fetch news headlines for this stock.")
