import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- UI ----------------

st.set_page_config(page_title="Bond Yield Forecasting", layout="wide")

st.title("AI Driven Bond Yield Forecasting")
st.subheader("Comparative Study of Forecasting Models")

# ---------------- METRICS ----------------

def directional_accuracy(actual, predicted):

    actual_diff = np.sign(np.diff(actual))
    pred_diff = np.sign(np.diff(predicted))

    return np.mean(actual_diff == pred_diff) * 100


def compute_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    da = directional_accuracy(actual, predicted)

    return mae, mse, da


# ---------------- GRAPH ----------------

def plot_graph(series, future_dates, forecast, title):

    plt.figure(figsize=(10,4))

    plt.plot(series.index, series.values, label="Actual")

    plt.plot(future_dates, forecast, label="Predicted")

    plt.title(title)

    plt.xlabel("Date")
    plt.ylabel("Bond Yield")

    plt.legend()

    st.pyplot(plt)


# ---------------- FILE UPLOAD ----------------

file = st.file_uploader("Upload Bond Yield CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    df.columns = [c.strip().upper() for c in df.columns]

    df["DATE"] = pd.to_datetime(df["DATE"])

    bonds = [c for c in df.columns if c != "DATE"]

    bond = st.selectbox("Select Bond Maturity", bonds)

    df = df[["DATE", bond]]

    df.rename(columns={bond:"YIELD"}, inplace=True)

    df.dropna(inplace=True)

    df.set_index("DATE", inplace=True)

    series = df["YIELD"]

    st.write("Dataset Preview")
    st.dataframe(df.head())

    forecast_days = st.number_input("Forecast Days", 1, 1000, 30)

    if st.button("Run Forecast"):

        last_date = series.index[-1]

        future_dates = pd.date_range(
            last_date,
            periods=forecast_days+1,
            freq="D"
        )[1:]

        results = []

# =====================================================
# 1 ARIMA
# =====================================================

        arima = ARIMA(series, order=(5,1,0))
        arima_fit = arima.fit()

        arima_forecast = arima_fit.forecast(steps=forecast_days)

        mae, mse, da = compute_metrics(
            series[-forecast_days:].values,
            arima_forecast[:forecast_days].values
        )

        st.header("ARIMA Forecast")

        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("Directional Accuracy:", da)

        plot_graph(series, future_dates, arima_forecast.values,
                   "ARIMA Bond Yield Forecast")

        results.append(["ARIMA", mae, mse, da])

# =====================================================
# 2 PCA MODEL
# =====================================================

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(series.values.reshape(-1,1))

        pca = PCA(n_components=1)

        pca_values = pca.fit_transform(scaled)

        X = pca_values[:-1]
        y = series.values[1:]

        model = LinearRegression()

        model.fit(X, y)

        last_val = pca_values[-1].reshape(1,-1)

        pca_forecast = []

        for i in range(forecast_days):

            pred = model.predict(last_val)[0]

            pca_forecast.append(pred)

        pca_forecast = np.array(pca_forecast)

        mae, mse, da = compute_metrics(
            series[-forecast_days:].values,
            pca_forecast[:forecast_days]
        )

        st.header("PCA Regression Forecast")

        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("Directional Accuracy:", da)

        plot_graph(series, future_dates, pca_forecast,
                   "PCA Bond Yield Forecast")

        results.append(["PCA Regression", mae, mse, da])

# =====================================================
# 3 LSTM MODEL
# =====================================================

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(series.values.reshape(-1,1))

        seq = 20

        X = []
        y = []

        for i in range(seq, len(scaled)):

            X.append(scaled[i-seq:i])
            y.append(scaled[i])

        X = np.array(X)
        y = np.array(y)

        model = Sequential()

        model.add(LSTM(32, input_shape=(seq,1)))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        model.fit(X, y, epochs=10, verbose=0)

        last_seq = scaled[-seq:]

        lstm_forecast = []

        current = last_seq.copy()

        for i in range(forecast_days):

            pred = model.predict(current.reshape(1,seq,1))[0]

            lstm_forecast.append(pred)

            current = np.append(current[1:], pred)

        lstm_forecast = scaler.inverse_transform(
            np.array(lstm_forecast).reshape(-1,1)
        ).flatten()

        mae, mse, da = compute_metrics(
            series[-forecast_days:].values,
            lstm_forecast[:forecast_days]
        )

        st.header("LSTM Forecast")

        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("Directional Accuracy:", da)

        plot_graph(series, future_dates, lstm_forecast,
                   "LSTM Bond Yield Forecast")

        results.append(["LSTM", mae, mse, da])

# =====================================================
# 4 MULTIVARIATE MODEL
# =====================================================

        df_lag = df.copy()

        for lag in range(1,4):
            df_lag[f"LAG{lag}"] = df_lag["YIELD"].shift(lag)

        df_lag.dropna(inplace=True)

        X = df_lag.drop("YIELD", axis=1)
        y = df_lag["YIELD"]

        model = LinearRegression()

        model.fit(X, y)

        last_row = X.iloc[-1].values.reshape(1,-1)

        multi_forecast = []

        for i in range(forecast_days):

            pred = model.predict(last_row)[0]

            multi_forecast.append(pred)

        multi_forecast = np.array(multi_forecast)

        mae, mse, da = compute_metrics(
            series[-forecast_days:].values,
            multi_forecast[:forecast_days]
        )

        st.header("Multivariate Regression Forecast")

        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("Directional Accuracy:", da)

        plot_graph(series, future_dates, multi_forecast,
                   "Multivariate Bond Yield Forecast")

        results.append(["Multivariate", mae, mse, da])

# =====================================================
# 5 HYBRID MODEL
# =====================================================

        residuals = series.values - arima_fit.predict().values

        scaler = MinMaxScaler()

        scaled_res = scaler.fit_transform(
            residuals.reshape(-1,1)
        )

        seq = 20

        X = []
        y = []

        for i in range(seq, len(scaled_res)):

            X.append(scaled_res[i-seq:i])
            y.append(scaled_res[i])

        X = np.array(X)
        y = np.array(y)

        model = Sequential()

        model.add(LSTM(32, input_shape=(seq,1)))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        model.fit(X, y, epochs=10, verbose=0)

        last_seq = scaled_res[-seq:]

        res_forecast = []

        current = last_seq.copy()

        for i in range(forecast_days):

            pred = model.predict(current.reshape(1,seq,1))[0]

            res_forecast.append(pred)

            current = np.append(current[1:], pred)

        res_forecast = scaler.inverse_transform(
            np.array(res_forecast).reshape(-1,1)
        ).flatten()

        hybrid_forecast = arima_forecast.values + res_forecast

        mae, mse, da = compute_metrics(
            series[-forecast_days:].values,
            hybrid_forecast[:forecast_days]
        )

        st.header("Hybrid ARIMA + LSTM Forecast")

        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("Directional Accuracy:", da)

        plot_graph(series, future_dates, hybrid_forecast,
                   "Hybrid Bond Yield Forecast")

        results.append(["Hybrid ARIMA + LSTM", mae, mse, da])

# =====================================================
# COMPARISON TABLE
# =====================================================

        st.header("Model Comparison")

        table = pd.DataFrame(
            results,
            columns=[
                "Model",
                "MAE",
                "MSE",
                "Directional Accuracy"
            ]
        )

        st.dataframe(table)
