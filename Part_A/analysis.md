# Part A — Model Comparison Analysis

**Dataset:** Daily minimum temperatures, Melbourne, Australia, 1981–1990 (3,650 daily observations).

## Results

| Model   | MAE  | RMSE | MAPE (%) |
|---------|------|------|----------|
| ARIMA(3,0,1) | 3.31 | 4.02 | 45.49 |
| Prophet | 2.36 | 2.90 | 32.09 |

Prophet outperformed ARIMA on every metric, cutting MAE by about 29% and MAPE by roughly 13
percentage points.

## Why Prophet won

The Augmented Dickey-Fuller test showed the raw series is already stationary (p ≈ 0.0002), so
`auto_arima` correctly chose a non-differenced order, ARIMA(3,0,1). That model captures the
short-term autocorrelation in the data well — today's temperature is a strong predictor of
tomorrow's — but it has no explicit mechanism for the fact that Melbourne's temperature swings
through a full summer-to-winter cycle every 365 days. Once the forecast horizon is long (the test
set here is the last 20% of the data, 730 days), an ARIMA model without a seasonal term
effectively reverts toward the historical mean and can't "remember" that day 400 of the forecast
should be warm because it falls in the Southern Hemisphere summer.

Prophet is built to decompose a series into trend + seasonality + (optional) holiday effects.
With `yearly_seasonality=True`, it explicitly fits a Fourier-series component that repeats every
year, so its 730-day-ahead forecast keeps tracking the seasonal cycle instead of flattening out.
The component plot confirms this: Prophet's yearly seasonality curve closely mirrors the
summer/winter pattern visible in the monthly-averaged raw data.

## Takeaway

For short-horizon forecasting with strong local autocorrelation and no clean seasonal cycle,
ARIMA (or SARIMA with a seasonal term) is a fine, interpretable choice. For a series with a clear
annual cycle and a long forecast horizon, an explicitly seasonal model such as Prophet is the
better fit — and this comparison shows why plotting the seasonal decomposition before picking a
model matters more than the model's reputation.
