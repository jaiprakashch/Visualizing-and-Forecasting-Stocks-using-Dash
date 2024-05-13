import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go

def lstm_prediction(stock, n_days):
    # Download stock data for last 60 days
    df = yf.download(stock, period='60d')
    df.reset_index(inplace=True)

    # Normalize the closing prices using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Prepare the input for LSTM
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 5
    X, y = create_dataset(scaled_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and testing sets
    training_size = int(len(X) * 0.85)
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    # Build the LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(100, return_sequences=False),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

    # Forecasting future days
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)

    last_batch = scaled_data[-time_step:]
    last_batch = last_batch.reshape((1, time_step, 1))
    future_predictions = []

    for i in range(n_days):
        current_pred = model.predict(last_batch)[0]
        future_predictions.append(current_pred)
        last_batch = np.append(last_batch[:, 1:, :], [[current_pred]], axis=1)

    future_predictions = scaler.inverse_transform(future_predictions)

    future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, n_days+1)]

    # Plot the results using Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df['Date'][-len(test_predictions):], 
            y=test_predictions.flatten(),
            mode='lines',
            name='Test Predictions'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates, 
            y=future_predictions.flatten(),
            mode='lines+markers',
            name='Future Predictions'
        )
    )
    fig.update_layout(
        title=f"Forecasted Close Prices for the Next {n_days} Days for {stock}",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )

    return fig
