import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler

model = load_model('/Users/ganish/Downloads/Python Local Files/Stock-Price-Predictor-via-LSTM/Stock Prediction Model.keras')

st.header('Stock Market Predictor')

#set input data (Stock name and timeframe to average over)
stock = st.text_input('Enter Stock Symbol', 'AAPL')
timeframe = int(st.text_input('Enter timeframe for averaging', 100))
start = '2014-01-01'
end = str(date.today())

#download data
data = yf.download(stock , start, end)
st.subheader('Stock Data')
st.write(data)

#split the data and transform it to a range of 0-1 so that the data can be 
#trained and tested
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data) - 1])
scaler = MinMaxScaler(feature_range=(0, 1))
past_timeframe_days = data_train.tail(timeframe)
data_test = pd.concat([past_timeframe_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

st.subheader(f'Moving Average of {timeframe} days')
ma_timeframe_days = data.Close.rolling(timeframe).mean()
fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_timeframe_days, 'r', label = f'Moving Average of {timeframe} days')
plt.plot(data.Close, 'g', label = 'Closing Stock Price')
plt.xlabel('Time (days)')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)

for i in range(timeframe, data_test_scale.shape[0]):
    x.append(data_test_scale[i - timeframe: i])
    y.append(data_test_scale[i, 0])
x, y = np.array(x), np.array(y)

#predict the closing stock price over the timeframe
predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

st.subheader("Original vs Predicted Stock Price")
fig2 = plt.figure(figsize = (10, 8))
plt.plot(predict, 'r', label = 'Predicted Stock Price')
plt.plot(y, 'g', label = 'Actual Stock Price')
plt.xlabel('Time (days)')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)


