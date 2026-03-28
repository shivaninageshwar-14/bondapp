import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(layout="wide")
st.title("📊 Bond Yield Advanced Multi-Model Dashboard")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("Upload Data")

tech_file = st.sidebar.file_uploader("Technical Indicators CSV", type=["csv"])
bond_file = st.sidebar.file_uploader("Bond Yield CSV", type=["csv"])

if tech_file is None or bond_file is None:
    st.warning("Upload both CSV files")
    st.stop()

# -------------------------------
# LOAD DATA
# -------------------------------
tech_df = pd.read_csv(tech_file)
bond_df = pd.read_csv(bond_file)

df = pd.merge(tech_df, bond_df, on="Date")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Ensure numeric + clean
df = df.apply(pd.to_numeric, errors='coerce')

# -------------------------------
# TARGET
# -------------------------------
TARGET = df.columns[0]  # ⚠️ change if needed

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["returns"] = df[TARGET].pct_change()
df["momentum"] = df[TARGET] - df[TARGET].shift(5)
df["volatility"] = df[TARGET].rolling(5).std()

df = df.dropna().reset_index(drop=True)

st.subheader("📄 Data Preview")
st.dataframe(df.head())

# -------------------------------
# METRICS
# -------------------------------
def compute_metrics(y_true, y_pred):

    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[-min_len:], y_pred[-min_len:]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    da = np.mean(
        np.sign(y_true[1:] - y_true[:-1]) ==
        np.sign(y_pred[1:] - y_pred[:-1])
    )

    return mse, rmse, da, y_true, y_pred

# -------------------------------
# SEQUENCE (FIXED)
# -------------------------------
def create_sequences(data, seq_len=10):

    data = np.array(data)

    if len(data.shape) == 1:
        data = data.reshape(-1,1)

    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])

    return np.array(X), np.array(y)

# -------------------------------
# ARIMA
# -------------------------------
def run_arima(series):

    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()

    pred = model_fit.forecast(steps=len(test))

    return test.values, pred.values

# -------------------------------
# LSTM (FIXED)
# -------------------------------
def run_lstm(series):

    series = np.array(series)

    if len(series.shape) == 1:
        series = series.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    seq_len = 10
    X, y = create_sequences(scaled, seq_len)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len,1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, verbose=0)

    pred = model.predict(X_test)

    # inverse scaling
    temp_pred = np.zeros((len(pred), 1))
    temp_pred[:,0] = pred.flatten()

    temp_true = np.zeros((len(y_test), 1))
    temp_true[:,0] = y_test.flatten()

    pred_inv = scaler.inverse_transform(temp_pred)
    y_test_inv = scaler.inverse_transform(temp_true)

    return y_test_inv.flatten(), pred_inv.flatten()

# -------------------------------
# HYBRID
# -------------------------------
def run_hybrid(series):

    y_true_arima, arima_pred = run_arima(series)

    residuals = y_true_arima - arima_pred

    scaler = MinMaxScaler()
    res_scaled = scaler.fit_transform(residuals.reshape(-1,1))

    seq_len = 5
    X, y = create_sequences(res_scaled, seq_len)

    if len(X) < 20:
        return y_true_arima, arima_pred

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(32, input_shape=(seq_len,1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, verbose=0)

    res_pred = model.predict(X_test)
    res_pred = scaler.inverse_transform(res_pred).flatten()

    min_len = min(len(arima_pred), len(res_pred))

    hybrid_pred = arima_pred[-min_len:] + res_pred[-min_len:]
    y_true = y_true_arima[-min_len:]

    return y_true, hybrid_pred

# -------------------------------
# TWO-HEAD LSTM (MULTI-TASK)
# -------------------------------
def run_two_head_lstm(df):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    seq_len = 10
    X, y = create_sequences(scaled, seq_len)

    y_direction = (np.diff(df[TARGET].values) > 0).astype(int)
    y_direction = y_direction[seq_len:]

    X1 = X[:,:,0].reshape(X.shape[0], seq_len, 1)
    X2 = X

    split = int(len(X) * 0.8)

    X1_train, X1_test = X1[:split], X1[split:]
    X2_train, X2_test = X2[:split], X2[split:]

    y_train, y_test = y[:split], y[split:]
    y_train_dir, y_test_dir = y_direction[:split], y_direction[split:]

    input1 = Input(shape=(seq_len,1))
    head1 = LSTM(32)(input1)

    input2 = Input(shape=(seq_len, X.shape[2]))
    head2 = LSTM(64)(input2)

    merged = Concatenate()([head1, head2])

    reg_output = Dense(1, name="yield_output")(merged)
    dir_output = Dense(1, activation='sigmoid', name="direction_output")(merged)

    model = Model(inputs=[input1, input2], outputs=[reg_output, dir_output])

    model.compile(
        optimizer='adam',
        loss={
            "yield_output": "mse",
            "direction_output": "binary_crossentropy"
        },
        loss_weights={
            "yield_output": 0.5,
            "direction_output": 0.5
        }
    )

    model.fit(
        [X1_train, X2_train],
        {
            "yield_output": y_train,
            "direction_output": y_train_dir
        },
        epochs=5,
        verbose=0
    )

    pred_value, pred_dir = model.predict([X1_test, X2_test])

    min_len = min(len(pred_value), len(y_test))

    pred_value = pred_value[-min_len:]
    y_test = y_test[-min_len:]
    pred_dir = pred_dir[-min_len:]

    temp_pred = np.zeros((min_len, df.shape[1]))
    temp_pred[:,0] = pred_value.flatten()

    temp_true = np.zeros((min_len, df.shape[1]))
    temp_true[:,0] = y_test.flatten()

    pred_inv = scaler.inverse_transform(temp_pred)[:,0]
    y_test_inv = scaler.inverse_transform(temp_true)[:,0]

    return y_test_inv, pred_inv

# -------------------------------
# RUN MODELS
# -------------------------------
models = {
    "ARIMA": lambda df: run_arima(df[TARGET]),
    "LSTM": lambda df: run_lstm(df[TARGET]),
    "Hybrid (ARIMA + LSTM)": lambda df: run_hybrid(df[TARGET]),
    "Two-Head LSTM (Directional)": run_two_head_lstm
}

results = []

st.subheader("📊 Model Outputs")

for name, func in models.items():

    try:
        y_true, y_pred = func(df)

        mse, rmse, da, y_true, y_pred = compute_metrics(y_true, y_pred)

        results.append([name, mse, rmse, da])

        st.subheader(f"{name} - Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_true, label="Actual")
        ax.plot(y_pred, label="Predicted")
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"{name} failed: {e}")

# -------------------------------
# RESULTS TABLE
# -------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "MSE", "RMSE", "Directional Accuracy"]
)

st.subheader("📋 Model Comparison Table")
st.dataframe(results_df)

# -------------------------------
# BEST MODEL
# -------------------------------
if not results_df.empty:
    best = results_df.sort_values("RMSE").iloc[0]

    st.success(f"""
🏆 Best Model: {best['Model']}

RMSE: {best['RMSE']:.4f}  
Directional Accuracy: {best['Directional Accuracy']*100:.2f}%
""")
