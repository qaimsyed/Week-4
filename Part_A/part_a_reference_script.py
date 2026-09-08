"""
Part A - Time Series Forecasting: ARIMA vs Prophet
Dataset: Daily Minimum Temperatures in Melbourne, Australia (1981-1990)
Source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

PLOTS = "plots"

# ---------------------------------------------------------------
# A2 - Load data + EDA
# ---------------------------------------------------------------
df = pd.read_csv("data/daily-min-temperatures.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

print(f"Rows: {len(df)}  |  Range: {df.ds.min().date()} to {df.ds.max().date()}")

# Plot the raw series
plt.figure(figsize=(12, 4))
plt.plot(df["ds"], df["y"], linewidth=0.6)
plt.title("Daily Minimum Temperature - Melbourne, Australia (1981-1990)")
plt.xlabel("Date"); plt.ylabel("Temp (C)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_raw_series.png", dpi=120)
plt.close()

# Trend/seasonality: resample monthly to see the pattern more clearly
monthly = df.set_index("ds")["y"].resample("MS").mean()
plt.figure(figsize=(12, 4))
monthly.plot()
plt.title("Monthly Average Minimum Temperature (shows yearly seasonality)")
plt.xlabel("Date"); plt.ylabel("Temp (C)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_monthly_seasonality.png", dpi=120)
plt.close()

# Augmented Dickey-Fuller test for stationarity
adf_result = adfuller(df["y"])
adf_summary = (
    f"ADF Statistic: {adf_result[0]:.4f}\n"
    f"p-value: {adf_result[1]:.6f}\n"
    f"Critical Values: {adf_result[4]}\n"
    f"Conclusion: {'Series is STATIONARY (reject H0)' if adf_result[1] < 0.05 else 'Series is NON-STATIONARY (fail to reject H0)'}"
)
print(adf_summary)
with open("adf_test_result.txt", "w") as f:
    f.write(adf_summary)

# ACF / PACF plots
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
plot_acf(df["y"], lags=40, ax=axes[0])
plot_pacf(df["y"], lags=40, ax=axes[1], method="ywm")
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_acf_pacf.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# A3 - Train/test split + ARIMA
# ---------------------------------------------------------------
test_size = int(len(df) * 0.20)
train, test = df.iloc[:-test_size], df.iloc[-test_size:]
print(f"Train size: {len(train)}, Test size: {len(test)}")

# The ADF test says the raw series is already stationary (temperature
# oscillates around a seasonal mean rather than drifting), so d is
# expected to come out as 0 or 1. We let auto_arima confirm this by
# searching over p, d, q instead of guessing by hand.
auto_model = pm.auto_arima(
    train["y"],
    start_p=0, start_q=0, max_p=5, max_q=5, d=None,
    seasonal=False, trace=False, stepwise=True,
    suppress_warnings=True,
)
order = auto_model.order
print(f"auto_arima selected order: {order}")

arima_model = ARIMA(train["y"].values, order=order).fit()
arima_forecast_res = arima_model.get_forecast(steps=len(test))
arima_pred = arima_forecast_res.predicted_mean
arima_ci = arima_forecast_res.conf_int(alpha=0.05)

with open("arima_summary.txt", "w") as f:
    f.write(f"Selected order (p,d,q) = {order}\n\n")
    f.write(str(arima_model.summary()))

# ---------------------------------------------------------------
# A4 - Prophet
# ---------------------------------------------------------------
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(train[["ds", "y"]])
future = prophet_model.make_future_dataframe(periods=len(test), freq="D")
prophet_forecast = prophet_model.predict(future)
prophet_pred = prophet_forecast["yhat"].iloc[-len(test):].values
prophet_lower = prophet_forecast["yhat_lower"].iloc[-len(test):].values
prophet_upper = prophet_forecast["yhat_upper"].iloc[-len(test):].values

# ---------------------------------------------------------------
# A5 - Metrics + comparison table
# ---------------------------------------------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true = test["y"].values

results = pd.DataFrame({
    "Model": ["ARIMA", "Prophet"],
    "MAE": [
        mean_absolute_error(y_true, arima_pred),
        mean_absolute_error(y_true, prophet_pred),
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_true, arima_pred)),
        np.sqrt(mean_squared_error(y_true, prophet_pred)),
    ],
    "MAPE (%)": [
        mape(y_true, arima_pred),
        mape(y_true, prophet_pred),
    ],
})
results.to_csv("comparison_table.csv", index=False)
print(results)

# ---------------------------------------------------------------
# Forecast plots: actual vs predicted with confidence intervals
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(test["ds"], y_true, label="Actual", color="black", linewidth=1)
axes[0].plot(test["ds"], arima_pred, label="ARIMA forecast", color="tab:blue")
axes[0].fill_between(test["ds"], arima_ci[:, 0], arima_ci[:, 1], color="tab:blue", alpha=0.2, label="95% CI")
axes[0].set_title(f"ARIMA{order} - Actual vs Forecast")
axes[0].legend()

axes[1].plot(test["ds"], y_true, label="Actual", color="black", linewidth=1)
axes[1].plot(test["ds"], prophet_pred, label="Prophet forecast", color="tab:orange")
axes[1].fill_between(test["ds"], prophet_lower, prophet_upper, color="tab:orange", alpha=0.2, label="95% CI")
axes[1].set_title("Prophet - Actual vs Forecast")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{PLOTS}/04_forecast_comparison.png", dpi=120)
plt.close()

# Prophet's own component plot (trend + yearly seasonality)
fig2 = prophet_model.plot_components(prophet_forecast)
fig2.savefig(f"{PLOTS}/05_prophet_components.png", dpi=120)
plt.close(fig2)

print("Part A complete. Outputs written to plots/, comparison_table.csv, arima_summary.txt, adf_test_result.txt")
