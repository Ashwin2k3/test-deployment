import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
    }
    h1, h2, h3 {
        color: #333;
    }
    .css-1d391kg {
        background-color: #e9ecef;
    }
    .css-1e7a8h1 {
        background-color: #f7f7f7;
    }
    .stMarkdown {
        color: #555;
    }
    .css-16huue1 {
        color: #007bff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache
def get_data():
    path = 'stock.csv'
    return pd.read_csv(path, low_memory=False)

df = get_data()
df = df.drop_duplicates(subset="Name", keep="first")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App title and introduction
st.title("📈 Stock Prediction App")
st.write("""
    Welcome to the Stock Prediction App! 
    This application provides forecasts of stock prices based on historical data.
    Please select a stock and choose the number of years for prediction.
""")

# Sidebar for user inputs
st.sidebar.header("Settings")
#st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)
stocks = df['Name']
selected_stock = st.sidebar.selectbox("Select Stock", stocks)

index = df[df["Name"]==selected_stock].index.values[0]
symbol = df["Symbol"][index]

n_years = st.sidebar.slider("Years of Prediction", 1, 5, 1)
period = n_years * 365

# Data loading
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(symbol)
data_load_state.text("Loading data... Done!")

st.write("##")

# Display raw data
st.subheader("Raw Data")
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Stock Open', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Stock Close', line=dict(color='#ff7f0e')))
    fig.update_layout(
        title="Time Series Data",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train = df_train.dropna()
df_train['ds'] = pd.to_datetime(df_train['ds'])

# Check for sufficient data
if df_train.shape[0] < 2:
    st.error("Not enough data to fit the model. Please ensure you have at least 2 non-NaN rows of data.")
else:
    m = Prophet()
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.write("###")
    st.subheader("Forecast Data")
    st.write(forecast.tail())

    # Plot forecast
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # Plot forecast components
    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    fig2 = m.plot_components(forecast)
    st.write(fig2)

