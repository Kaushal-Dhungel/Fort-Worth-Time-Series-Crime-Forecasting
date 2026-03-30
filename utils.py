from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX


CRIME_PATH = "Data/Cleaned/CFW_Monthly_Crime_Count_Cleaned.csv"
TEMP_PATH = "Data/Cleaned/CFW_Avg_Tempr_Cleaned.csv"
TRAIN_END = "2024-12-01"
FORECAST_STEPS = 12
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


@dataclass
class ModelBundle:
    label: str
    series_name: str
    result: object
    forecast: pd.DataFrame
    fitted: pd.Series
    residuals: pd.Series
    metrics: dict[str, float]
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


def _prepare_monthly_index(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    data = data.drop_duplicates(subset=["Date"])
    data = data.set_index("Date").asfreq("MS")
    return data[[value_column]]


def load_crime_data() -> pd.DataFrame:
    crime = pd.read_csv(CRIME_PATH)
    return _prepare_monthly_index(crime, "Crime_Count")


def load_temperature_data() -> pd.DataFrame:
    temp = pd.read_csv(TEMP_PATH)
    return _prepare_monthly_index(temp, "Temp")


def prepare_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crime = load_crime_data()
    temp = load_temperature_data()
    merged = crime.join(temp, how="inner")
    merged["Crime_3M"] = merged["Crime_Count"].rolling(3).mean()
    merged["Temp_3M"] = merged["Temp"].rolling(3).mean()
    return crime, temp, merged


def dataset_summary(crime: pd.DataFrame, temp: pd.DataFrame, merged: pd.DataFrame) -> dict[str, object]:
    return {
        "crime_rows": len(crime),
        "temp_rows": len(temp),
        "merged_rows": len(merged),
        "crime_start": crime.index.min(),
        "crime_end": crime.index.max(),
        "temp_start": temp.index.min(),
        "temp_end": temp.index.max(),
        "merged_start": merged.index.min(),
        "merged_end": merged.index.max(),
        "crime_mean": float(crime["Crime_Count"].mean()),
        "crime_max": float(crime["Crime_Count"].max()),
        "crime_min": float(crime["Crime_Count"].min()),
        "temp_mean": float(merged["Temp"].mean()),
        "correlation": float(merged["Crime_Count"].corr(merged["Temp"])),
    }


def monthly_profile(crime: pd.DataFrame) -> pd.DataFrame:
    profile = crime.reset_index().copy()
    profile["Month"] = pd.Categorical(
        profile["Date"].dt.month_name(),
        categories=MONTH_ORDER,
        ordered=True,
    )
    result = (
        profile.groupby("Month", observed=False)["Crime_Count"]
        .mean()
        .reset_index(name="Average_Crime_Count")
    )
    return result


def add_year_month_fields(crime: pd.DataFrame) -> pd.DataFrame:
    data = crime.reset_index().copy()
    data["Year"] = data["Date"].dt.year.astype(str)
    data["Month"] = pd.Categorical(
        data["Date"].dt.month_name(),
        categories=MONTH_ORDER,
        ordered=True,
    )
    return data


def stl_decomposition(crime: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    stl = STL(crime["Crime_Count"], period=period, robust=True).fit()
    return pd.DataFrame(
        {
            "Observed": stl.observed,
            "Trend": stl.trend,
            "Seasonal": stl.seasonal,
            "Residual": stl.resid,
        },
        index=crime.index,
    )


def split_train_test(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = merged.loc[:TRAIN_END].copy()
    test = merged.loc[pd.Timestamp(TRAIN_END) + pd.offsets.MonthBegin(1) :].copy()
    return train, test


def metric_summary(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mape = (np.abs((actual - predicted) / actual).mean()) * 100
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def _future_month_index(last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS")


def fit_baseline_sarima(
    merged: pd.DataFrame,
    order: tuple[int, int, int] = SARIMA_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMA_SEASONAL_ORDER,
) -> ModelBundle:
    train, test = split_train_test(merged)
    model = SARIMAX(
        train["Crime_Count"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    fitted = pd.Series(result.fittedvalues, index=train.index, name="Fitted")
    forecast_obj = result.get_forecast(steps=len(test))
    forecast = pd.DataFrame(
        {
            "Actual": test["Crime_Count"],
            "Forecast": forecast_obj.predicted_mean,
            "Lower": forecast_obj.conf_int().iloc[:, 0],
            "Upper": forecast_obj.conf_int().iloc[:, 1],
        },
        index=test.index,
    )
    metrics = metric_summary(forecast["Actual"], forecast["Forecast"])
    return ModelBundle(
        label="SARIMA Baseline",
        series_name="Crime_Count",
        result=result,
        forecast=forecast,
        fitted=fitted,
        residuals=result.resid,
        metrics=metrics,
        order=order,
        seasonal_order=seasonal_order,
    )


def fit_arimax_with_temperature(
    merged: pd.DataFrame,
    order: tuple[int, int, int] = SARIMA_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMA_SEASONAL_ORDER,
    future_steps: int = FORECAST_STEPS,
) -> tuple[ModelBundle, pd.DataFrame]:
    train, test = split_train_test(merged)
    model = SARIMAX(
        train["Crime_Count"],
        exog=train[["Temp"]],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    fitted = pd.Series(result.fittedvalues, index=train.index, name="Fitted")
    holdout_obj = result.get_forecast(steps=len(test), exog=test[["Temp"]])
    holdout = pd.DataFrame(
        {
            "Actual": test["Crime_Count"],
            "Forecast": holdout_obj.predicted_mean,
            "Lower": holdout_obj.conf_int().iloc[:, 0],
            "Upper": holdout_obj.conf_int().iloc[:, 1],
            "Temp": test["Temp"],
        },
        index=test.index,
    )
    metrics = metric_summary(holdout["Actual"], holdout["Forecast"])

    temp_model = ExponentialSmoothing(
        merged["Temp"],
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)
    future_index = _future_month_index(merged.index.max(), future_steps)
    future_temp = pd.Series(temp_model.forecast(future_steps), index=future_index, name="Temp")

    full_model = SARIMAX(
        merged["Crime_Count"],
        exog=merged[["Temp"]],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    full_result = full_model.fit(disp=False)
    future_obj = full_result.get_forecast(steps=future_steps, exog=future_temp.to_frame())
    future_forecast = pd.DataFrame(
        {
            "Forecast": future_obj.predicted_mean,
            "Lower": future_obj.conf_int().iloc[:, 0],
            "Upper": future_obj.conf_int().iloc[:, 1],
            "Projected_Temp": future_temp,
        },
        index=future_index,
    )

    bundle = ModelBundle(
        label="ARIMAX With Temperature",
        series_name="Crime_Count",
        result=result,
        forecast=holdout,
        fitted=fitted,
        residuals=result.resid,
        metrics=metrics,
        order=order,
        seasonal_order=seasonal_order,
    )
    return bundle, future_forecast


def combined_forecast_frame(history: pd.DataFrame, future_forecast: pd.DataFrame) -> pd.DataFrame:
    history_frame = history[["Crime_Count"]].rename(columns={"Crime_Count": "Observed"})
    future = future_forecast.rename(columns={"Forecast": "Projected"})
    return history_frame.join(future, how="outer")


def residual_diagnostics_frame(residuals: pd.Series) -> pd.DataFrame:
    resid = residuals.dropna()
    return pd.DataFrame(
        {
            "Residual": resid,
            "Abs_Residual": resid.abs(),
        }
    )


def acf_pacf_figure(series: pd.Series, lags: int = 24):
    import matplotlib.pyplot as plt

    clean = series.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(clean, ax=axes[0], lags=lags)
    plot_pacf(clean, ax=axes[1], lags=lags, method="ywm")
    axes[0].set_title("Residual Autocorrelation")
    axes[1].set_title("Residual Partial Autocorrelation")
    fig.tight_layout()
    return fig
