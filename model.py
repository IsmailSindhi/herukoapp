import yfinance as yf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from pandas_datareader import data as pdr
import datetime as dt

yf.pdr_override()
# 'BTC-USD'
def model(crypto, days):
    end_date = dt.datetime.today()
    start_date = dt.datetime(2017,1,1)
    stock = crypto
    data = pdr.get_data_yahoo(stock, start_date, end_date)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    prediction_days = 60
    future_days = days

    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)-future_days):
        x_train.append(scaled_data[x-prediction_days:x,0])
        y_train.append(scaled_data[x+future_days, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    test_end = dt.datetime.now()
    test_start = dt.datetime(2020,1,1)


    test_data = pdr.get_data_yahoo(stock, test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    plt.plot(actual_prices, color = 'black', label='actual Prices')
    plt.plot(prediction_prices, color='green', label = 'predicted prices')
    plt.plot('crypto currency pice prediction')
    plt.xlabel('Time')
    plt.ylabel('price')
    plt.legend(loc='uppder left')
    plt.show()
    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs)+1,0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction


model('BTC-USD', 30)