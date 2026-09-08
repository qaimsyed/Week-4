# %% [markdown]
# # Part A — Time Series Forecasting: ARIMA vs Prophet
#
# **Dataset:** Daily minimum temperatures, Melbourne, Australia, 1981–1990
# (3,650 daily observations — well over the 500-observation / 2-year minimum).
# Source: [jbrownlee/Datasets](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv)
#
# **Goal:** build an ARIMA model and a Prophet model on the same data, compare
# their forecast accuracy, and explain *why* one wins.

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %% [markdown]
# ## A2 — Load data and explore
#
# We first plot the raw daily series, then a monthly average to make the
# yearly seasonal cycle easy to see, then formally test for stationarity.

# %%
df = pd.read_csv("data/daily-min-temperatures.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)
print(f"Rows: {len(df)} | Range: {df.ds.min().date()} to {df.ds.max().date()}")
df.head()

# %%
plt.figure(figsize=(12, 4))
plt.plot(df["ds"], df["y"], linewidth=0.6)
plt.title("Daily Minimum Temperature - Melbourne, Australia (1981-1990)")
plt.xlabel("Date"); plt.ylabel("Temp (C)")
plt.tight_layout()
plt.savefig("plots/01_raw_series.png", dpi=120)
plt.show()

# %%
monthly = df.set_index("ds")["y"].resample("MS").mean()
plt.figure(figsize=(12, 4))
monthly.plot()
plt.title("Monthly Average Minimum Temperature (yearly seasonality is clear here)")
plt.xlabel("Date"); plt.ylabel("Temp (C)")
plt.tight_layout()
plt.savefig("plots/02_monthly_seasonality.png", dpi=120)
plt.show()

# %% [markdown]
# ### Augmented Dickey-Fuller (ADF) test
#
# The ADF test checks whether a series is *stationary* — whether its mean and
# variance stay roughly constant over time. The null hypothesis (H0) is "the
# series has a unit root" (i.e. is non-stationary). A small p-value lets us
# reject H0 and conclude the series is stationary, which tells ARIMA's `d`
# (differencing order) whether it needs to difference the data at all.

# %%
adf_result = adfuller(df["y"])
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.6f}")
print(f"Critical values: {adf_result[4]}")
print("Conclusion:", "STATIONARY (reject H0)" if adf_result[1] < 0.05 else "NON-STATIONARY (fail to reject H0)")

# %% [markdown]
# ### ACF / PACF
#
# The autocorrelation function (ACF) and partial autocorrelation function
# (PACF) plots help pick starting values for ARIMA's `p` (AR order) and `q`
# (MA order): PACF cutting off sharply suggests an AR term of that lag, ACF
# cutting off sharply suggests an MA term. Here we just use them for
# intuition — `auto_arima` below does the formal search.

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
plot_acf(df["y"], lags=40, ax=axes[0])
plot_pacf(df["y"], lags=40, ax=axes[1], method="ywm")
plt.tight_layout()
plt.savefig("plots/03_acf_pacf.png", dpi=120)
plt.show()

# %% [markdown]
# ## A3 — Train/test split and ARIMA
#
# We hold out the last 20% of the series as a test set (no shuffling — time
# series must stay in chronological order) and let `auto_arima` search over
# (p, d, q) using AIC rather than guessing by hand.

# %%
test_size = int(len(df) * 0.20)
train, test = df.iloc[:-test_size], df.iloc[-test_size:]
print(f"Train size: {len(train)}, Test size: {len(test)}")

auto_model = pm.auto_arima(
    train["y"],
    start_p=0, start_q=0, max_p=5, max_q=5, d=None,
    seasonal=False, trace=False, stepwise=True,
    suppress_warnings=True,
)
order = auto_model.order
print(f"auto_arima selected order (p,d,q) = {order}")

arima_model = ARIMA(train["y"].values, order=order).fit()
arima_forecast_res = arima_model.get_forecast(steps=len(test))
arima_pred = arima_forecast_res.predicted_mean
arima_ci = arima_forecast_res.conf_int(alpha=0.05)
print(arima_model.summary())

# %% [markdown]
# ## A4 — Prophet
#
# Prophet decomposes the series into trend + seasonality (+ holidays, unused
# here) and fits them with a Bayesian curve-fitting approach rather than the
# linear autoregression ARIMA uses. We explicitly turn on yearly seasonality
# since that's the dominant pattern in temperature data, and turn off
# weekly/daily seasonality since neither is meaningful for this series.

# %%
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(train[["ds", "y"]])
future = prophet_model.make_future_dataframe(periods=len(test), freq="D")
prophet_forecast = prophet_model.predict(future)
prophet_pred = prophet_forecast["yhat"].iloc[-len(test):].values
prophet_lower = prophet_forecast["yhat_lower"].iloc[-len(test):].values
prophet_upper = prophet_forecast["yhat_upper"].iloc[-len(test):].values

# %% [markdown]
# ## A5 — Metrics and comparison table

# %%
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true = test["y"].values

results = pd.DataFrame({
    "Model": ["ARIMA", "Prophet"],
    "MAE": [mean_absolute_error(y_true, arima_pred), mean_absolute_error(y_true, prophet_pred)],
    "RMSE": [np.sqrt(mean_squared_error(y_true, arima_pred)), np.sqrt(mean_squared_error(y_true, prophet_pred))],
    "MAPE (%)": [mape(y_true, arima_pred), mape(y_true, prophet_pred)],
})
results.to_csv("comparison_table.csv", index=False)
results

# %% [markdown]
# ## Forecast plots: actual vs. predicted with 95% confidence intervals

# %%
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
plt.savefig("plots/04_forecast_comparison.png", dpi=120)
plt.show()

# %%
fig2 = prophet_model.plot_components(prophet_forecast)
fig2.savefig("plots/05_prophet_components.png", dpi=120)
plt.show()

# %% [markdown]
# ## Conclusion
#
# Prophet beat ARIMA on every metric (see `comparison_table.csv` and
# `analysis.md` for the full write-up). In short: the ADF test shows the raw
# series is already stationary, so `auto_arima` picked a non-seasonal
# ARIMA(3,0,1) — good at short-term autocorrelation but with no mechanism to
# track the yearly hot/cold cycle over a long forecast horizon. Prophet's
# explicit yearly-seasonality component keeps tracking that cycle 730 days
# out, which is why its error is roughly 30% lower across the board.
