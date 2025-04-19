!pip install streamlit yfinance scikit-learn plotly
# Need to type "streamlit run Stock_Prediction_app.py on terminal (This is being done on VScode)"
import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import plotly.graph_objs as go

st.title('Stock Market Prediction App (Random Forest Model & Rolling Average)')
# text box
selected_stock = st.text_input('Enter stock ticker:', value='NVDA', max_chars=10)

# Data scraping/cleaning 
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="max") # Might need to be fixed at 2010 and above to cut run time

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    if data.empty:
        st.error("No data downloaded. Check your stock ticker.")
        return pd.DataFrame()

    st.write("Downloaded columns:", data.columns.tolist())

    if "Close" not in data.columns:
        st.error("Missing 'Close' column.")
        return pd.DataFrame()

    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors='ignore')

    try:
        data["Tomorrow"] = data["Close"].shift(-1)

        # Align the Series before comparison
        tomorrow_aligned, close_aligned = data["Tomorrow"].align(data["Close"], axis=0)
        data["Target"] = (tomorrow_aligned > close_aligned).astype(int)

        data.dropna(subset=["Tomorrow"], inplace=True)

    except Exception as e:
        st.error(f"Error generating target columns: {e}")
        return pd.DataFrame()

    return data

data_load_state = st.text('Loading data...')
stock_data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

if stock_data.empty:
    st.stop()

st.subheader('Raw data')
st.write(stock_data.tail())

# Visual Graph
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name="Close"))
    fig.update_layout(title_text='Stock Prices Over Time', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Rolling Averages
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = stock_data.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    stock_data[ratio_column] = stock_data["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    stock_data[trend_column] = stock_data.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

stock_data.dropna(inplace=True)

# ML
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1 # 0.6 makes it so that we only count 1 if we are really sure it will be 1 (Going up)
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictions = backtest(stock_data, model, new_predictors)

# Count of 1 and 0 predicted
st.subheader('Model Evaluation')
st.write("Prediction counts:")
st.write(predictions["Predictions"].value_counts())

# Precision of the model
model_precision = precision_score(predictions["Target"], predictions["Predictions"])
st.write("Precision score (model):")
st.write(model_precision)

# Baseline accuracy (just guessing "up" every day)
baseline = predictions["Target"].value_counts(normalize=True)
st.write("Baseline: If you bought at open and sold at close every day:")
st.write(baseline)

# Compare model vs baseline
accuracy_gain = model_precision - baseline.get(1, 0)

# Determine color and wording based on gain
if accuracy_gain >= 0:
    color = "green"
    word = "more"
else:
    color = "red"
    word = "less"

st.markdown(
    f"<span style='color:{color}; font-size:14px;'>Using this prediction, you were {abs(accuracy_gain):.2%} {word} accurate than blindly predicting 'up'.</span>",
    unsafe_allow_html=True
)

# Predict tomorrow's movement
latest_prediction = predictions["Predictions"].iloc[-1]

if latest_prediction == 1:
    prediction_text = "up"
    prediction_color = "green"
    prediction_emoji = "📈"
else:
    prediction_text = "down"
    prediction_color = "red"
    prediction_emoji = "📉"

st.markdown(
    f"<span style='color:{prediction_color}; font-size:18px;'>{prediction_emoji}Based on the model, it is likely that the asset will go {prediction_text} tomorrow.</span>",
    unsafe_allow_html=True
)

st.write("---")
st.markdown(
     "Created by Matthew Lim |   [LinkedIn](https://www.linkedin.com/in/matthew-sj-lim/)"
)

