import sklearn.preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense

st.set_page_config(page_title="Milk Production Forecast", layout="centered")

st.title("📈 Milk Production Forecasting using GRU")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col='Month')
    df.index = pd.to_datetime(df.index)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Hyperparameters
    time_steps = st.slider("Time Steps (Sequence Length)", 3, 24, 12)
    neurons = st.slider("GRU Neurons", 10, 200, 100)
    epochs = st.slider("Epochs", 10, 500, 100)
    future_months = st.slider("Months to Forecast", 1, 24, 12)

    # Preprocessing
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    def create_sequences(data, time_steps):
        x, y = [], []
        for i in range(len(data) - time_steps):
            x.append(data[i:i+time_steps])
            y.append(data[i+time_steps])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(scaled_data, time_steps)

    # Model Definition
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(units=neurons, activation='relu', return_sequences=False, input_shape=(time_steps, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    with st.spinner("Training the model..."):
        model.fit(x_train, y_train, epochs=epochs, verbose=0)

    # Forecasting
    test_seed = scaled_data[-time_steps:]
    predictions = []

    for _ in range(future_months):
        input_seq = test_seed[-time_steps:].reshape(1, time_steps, 1)
        pred = model.predict(input_seq)[0][0]
        predictions.append(pred)
        test_seed = np.append(test_seed, [[pred]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=future_months, freq='MS')
    forecast_df = pd.DataFrame(predictions, index=forecast_dates, columns=['Predicted'])

    st.subheader("Forecasted Milk Production")
    st.dataframe(forecast_df)

    # Plot
    st.subheader("📊 Forecast Plot")
    fig, ax = plt.subplots()
    df.plot(ax=ax, label="Actual", legend=True)
    forecast_df.plot(ax=ax, label="Forecast", legend=True)
    st.pyplot(fig)
