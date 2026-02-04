# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI-Driven Bond Yield Forecasting", layout="wide")
st.title("AI-Driven Bond Yield Forecasting")
st.subheader("Hybrid ARIMA + Two-Head LSTM for Government Bond Yields")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Bond Yield CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Standardize column names
    df.columns = [c.strip().upper() for c in df.columns]

    if "DATE" not in df.columns:
        st.error("CSV must contain a DATE column")
        st.stop()

    # Identify bond columns (everything except DATE)
    bond_columns = [c for c in df.columns if c != "DATE"]

    if len(bond_columns) == 0:
        st.error("No bond yield columns found")
        st.stop()

    # -------- Bond Selection --------
    selected_bond = st.selectbox(
        "Select Bond Maturity",
        bond_columns
    )

    # -------- Data Cleaning --------
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[["DATE", selected_bond]]
    df.rename(columns={selected_bond: "YIELD"}, inplace=True)
    df.dropna(inplace=True)
    df.set_index("DATE", inplace=True)

    series = df["YIELD"]

    st.subheader("Data Preview (Selected Bond Only)")
    st.dataframe(df.head())

    n_days = st.number_input("Days to Forecast", 1, 365, 60)

    # ---------------- SUBMIT ----------------
    if st.button("Submit Forecast"):
        st.info("Training models and generating forecasts...")

        # -------- Train / Test Split --------
        split = int(len(series) * 0.8)
        train, test = series[:split], series[split:]

        # ================= ARIMA =================
        arima = ARIMA(train, order=(5,1,0))
        arima_fit = arima.fit()

        arima_test_pred = arima_fit.forecast(steps=len(test))
        arima_future = arima_fit.forecast(steps=n_days)

        # ================= RESIDUALS =================
        arima_train_pred = arima_fit.predict(
            start=train.index[0],
            end=train.index[-1]
        )

        residuals = train.values - arima_train_pred.values

        # ================= SCALING =================
        scaler = MinMaxScaler()
        residuals_scaled = scaler.fit_transform(residuals.reshape(-1,1))

        # ================= TWO-HEAD LSTM =================
        seq_len = 20

        X_level, X_diff, y = [], [], []

        for i in range(seq_len, len(residuals_scaled)):
            level = residuals_scaled[i-seq_len:i, 0]
            diff = np.diff(level, prepend=level[0])
            X_level.append(level)
            X_diff.append(diff)
            y.append(residuals_scaled[i, 0])

        X_level = np.array(X_level).reshape(-1, seq_len, 1)
        X_diff = np.array(X_diff).reshape(-1, seq_len, 1)
        y = np.array(y)

        # ---- Two LSTM Heads ----
        input_level = Input(shape=(seq_len,1))
        input_diff = Input(shape=(seq_len,1))

        lstm_level = LSTM(32)(input_level)
        lstm_diff = LSTM(32)(input_diff)

        merged = Dense(32, activation="relu")(lstm_level + lstm_diff)
        output = Dense(1)(merged)

        model = Model(inputs=[input_level, input_diff], outputs=output)
        model.compile(optimizer="adam", loss="mse")
        model.fit([X_level, X_diff], y, epochs=20, batch_size=16, verbose=0)

        # ================= FUTURE RESIDUALS =================
        last_level = residuals_scaled[-seq_len:].reshape(1, seq_len, 1)
        last_diff = np.diff(last_level[0,:,0], prepend=last_level[0,0,0]).reshape(1, seq_len, 1)

        future_res = []

        for _ in range(n_days):
            pred = model.predict([last_level, last_diff], verbose=0)[0][0]
            future_res.append(pred)

            last_level = np.append(last_level[:,1:,:], [[[pred]]], axis=1)
            last_diff = np.diff(last_level[0,:,0], prepend=last_level[0,0,0]).reshape(1, seq_len, 1)

        future_res = scaler.inverse_transform(
            np.array(future_res).reshape(-1,1)
        ).flatten()

        hybrid_future = arima_future.values + future_res

        # ================= METRICS =================
        mae = mean_absolute_error(test.values, arima_test_pred.values)
        rmse = np.sqrt(mean_squared_error(test.values, arima_test_pred.values))

        st.subheader("Accuracy (ARIMA Baseline)")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")

        # ================= DATES =================
        future_dates = pd.date_range(
            series.index[-1],
            periods=n_days+1,
            freq="D"
        )[1:]

        # ================= PLOTS =================
        st.subheader("ARIMA Forecast")
        plt.figure(figsize=(10,4))
        plt.plot(series.index, series.values, label="Historical")
        plt.plot(future_dates, arima_future.values, label="ARIMA Forecast")
        plt.legend()
        st.pyplot(plt)

        st.subheader("Two-Head LSTM (Hybrid) Forecast")
        plt.figure(figsize=(10,4))
        plt.plot(series.index, series.values, label="Historical")
        plt.plot(future_dates, hybrid_future, label="Hybrid Forecast")
        plt.legend()
        st.pyplot(plt)

        # ================= FINAL TABLE =================
        st.subheader("Forecast Results")
        result_df = pd.DataFrame({
            "Date": future_dates,
            "ARIMA Forecast": arima_future.values,
            "Two-Head LSTM Forecast": hybrid_future
        })
        st.dataframe(result_df)

        st.success("Forecast completed successfully!")