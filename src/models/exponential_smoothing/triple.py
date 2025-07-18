from numpy.typing import NDArray
import numpy as np
import pandas as pd

from evaluation.key_performance_indicators import kpi


def mul_seasonal_factors_initialization(
    seasonal_factors: NDArray,
    demand: NDArray,
    season_length: int,
    historical_length: int
) -> NDArray:
    """
    Initializes the seasonal factors based on the historical demand data.

    Parameters
    ----------
    seasonal_factors : NDArray
        Array to hold the seasonal factors.
    demand : NDArray
        Historical demand data.
    season_length : int
        Length of the seasonal cycle.
    historical_length : int
        Length of the historical demand data.

    Returns
    -------
    NDArray
        Updated seasonal factors.
    """
    for i in range(season_length):
        seasonal_factors[i] = np.mean(
            demand[i:historical_length:season_length]
        )  # Season average

    # Scale the seasonal factors (sum to season_length)
    seasonal_factors /= np.mean(seasonal_factors[:season_length])
    return seasonal_factors


def triple_exponential_smoothing_mul(
    demand: NDArray,
    alpha: float = 0.4,
    beta: float = 0.4,
    phi: float = 0.9,
    gamma: float = 0.3,
    season_length: int = 12,
    extra_periods: int = 1
) -> pd.DataFrame:
    """
    Computes a triple exponential smoothing forecast for demand data.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    alpha : float, optional
        Smoothing factor for level, by default 0.4.
    beta : float, optional
        Smoothing factor for trend, by default 0.4.
    phi : float, optional
        Damping factor for trend, by default 0.9.
    gamma : float, optional
        Smoothing factor for seasonality, by default 0.3.
    season_length : int, optional
        Length of the seasonal cycle, by default 12.
    extra_periods : int, optional
        Number of periods to forecast beyond the historical data, by default 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the historical demand, forecasted values,
        and the error (difference between demand and forecast).
    """
    # Historical period length
    historical_length = len(demand)
    # Append NaN values for extra periods
    demand = np.append(demand, [np.nan] * extra_periods)
    # Define the full forecast array
    forecast, a, b, s = np.full((4, historical_length + extra_periods), np.nan)
    s = mul_seasonal_factors_initialization(
        seasonal_factors=s,
        demand=demand,
        season_length=season_length,
        historical_length=historical_length
    )
    # Initialize the first forecast value
    a[0] = demand[0] / s[0]
    b[0] = demand[1] / s[1] - demand[0] / s[0]

    # Create the forecast for the first season
    for t in range(1, season_length):
        forecast[t] = (a[t - 1] + phi * b[t - 1]) * s[t]
        a[t] = alpha * (demand[t] / s[t]) + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]

    # Create the t+1 forecast until end of the historical period
    for t in range(season_length, historical_length):
        forecast[t] = (a[t - 1] + phi * b[t - 1]) * s[t - season_length]
        a[t] = alpha * (demand[t] / s[t - season_length]) + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]
        s[t] = gamma * (demand[t] / a[t]) + (1 - gamma) * s[t - season_length]

    # Create the t+1 forecast for the extra periods
    for t in range(historical_length, historical_length + extra_periods):
        forecast[t] = (a[t - 1] + phi * b[t - 1]) * s[t - season_length]
        a[t] = forecast[t] / s[t - season_length]
        b[t] = phi * b[t - 1]
        s[t] = s[t - season_length]

    forecast_df = pd.DataFrame.from_dict({
        "Demand": demand,
        "Forecast": forecast,
        "Level": a,
        "Trend": b,
        "Season": s,
        "Error": forecast - demand
    })
    forecast_df.index.name = "Period"
    return forecast_df


def add_seasonal_factors_initialization(
    seasonal_factors: NDArray,
    demand: NDArray,
    season_length: int,
    historical_length: int
) -> NDArray:
    """
    Initializes the seasonal factors based on the historical demand data.

    Parameters
    ----------
    seasonal_factors : NDArray
        Array to hold the seasonal factors.
    demand : NDArray
        Historical demand data.
    season_length : int
        Length of the seasonal cycle.
    historical_length : int
        Length of the historical demand data.

    Returns
    -------
    NDArray
        Updated seasonal factors.
    """
    for i in range(season_length):
        seasonal_factors[i] = np.mean(
            demand[i:historical_length:season_length]
        )  # Season average

    # Scale the seasonal factors (sum to 0)
    seasonal_factors -= np.mean(seasonal_factors[:season_length])
    return seasonal_factors


def triple_exponential_smoothing_add(
    demand: NDArray,
    alpha: float = 0.4,
    beta: float = 0.4,
    phi: float = 0.9,
    gamma: float = 0.3,
    season_length: int = 12,
    extra_periods: int = 1
) -> pd.DataFrame:
    """
    Computes a triple exponential smoothing forecast for demand data.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    alpha : float, optional
        Smoothing factor for level, by default 0.4.
    beta : float, optional
        Smoothing factor for trend, by default 0.4.
    phi : float, optional
        Damping factor for trend, by default 0.9.
    gamma : float, optional
        Smoothing factor for seasonality, by default 0.3.
    season_length : int, optional
        Length of the seasonal cycle, by default 12.
    extra_periods : int, optional
        Number of periods to forecast beyond the historical data, by default 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the historical demand, forecasted values,
        and the error (difference between demand and forecast).
    """
    # Historical period length
    historical_length = len(demand)
    # Append NaN values for extra periods
    demand = np.append(demand, [np.nan] * extra_periods)
    # Define the full forecast array
    forecast, a, b, s = np.full((4, historical_length + extra_periods), np.nan)
    s = add_seasonal_factors_initialization(
        seasonal_factors=s,
        demand=demand,
        season_length=season_length,
        historical_length=historical_length
    )
    # Initialize the first forecast value
    a[0] = demand[0] - s[0]
    b[0] = (demand[1] - s[1]) - (demand[0] - s[0])

    # Create the forecast for the first season
    for t in range(1, season_length):
        forecast[t] = (a[t - 1] + phi * b[t - 1]) + s[t]
        a[t] = alpha * (demand[t] - s[t]) + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]

    # Create the t+1 forecast until end of the historical period
    for t in range(season_length, historical_length):
        forecast[t] = (a[t - 1] + phi * b[t - 1]) + s[t - season_length]
        a[t] = alpha * (demand[t] - s[t - season_length]) + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]
        s[t] = gamma * (demand[t] - a[t]) + (1 - gamma) * s[t - season_length]

    # Create the t+1 forecast for the extra periods
    for t in range(historical_length, historical_length + extra_periods):
        forecast[t] = (a[t - 1] + phi * b[t - 1]) + s[t - season_length]
        a[t] = forecast[t] - s[t - season_length]
        b[t] = phi * b[t - 1]
        s[t] = s[t - season_length]

    forecast_df = pd.DataFrame.from_dict({
        "Demand": demand,
        "Forecast": forecast,
        "Level": a,
        "Trend": b,
        "Season": s,
        "Error": forecast - demand
    })
    forecast_df.index.name = "Period"
    return forecast_df


if __name__ == "__main__":
    demand = np.array([
        14, 10, 6, 2, 18, 8, 4, 1, 16, 9, 5, 3, 18, 11, 4, 2, 17, 9, 5, 1
    ])
    forecast_df = triple_exponential_smoothing_add(
        demand=demand,
        alpha=0.3,
        beta=0.2,
        phi=0.9,
        gamma=0.2,
        season_length=12,
        extra_periods=4
    )
    kpi(forecast_df=forecast_df)
    forecast_df[["Demand", "Forecast"]].plot(
        figsize=(8, 3), title="Double smoothing", ylim=(0, 30)
    )
