from numpy.typing import NDArray

import numpy as np
import pandas as pd


def double_exponential_smoothing(
    demand: NDArray,
    alpha: float = 0.4,
    beta: float = 0.4,
    extra_periods: int = 1
) -> pd.DataFrame:
    """
    Computes a double exponential smoothing forecast for demand data.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    alpha : float, optional
        Smoothing factor between 0 and 1, by default 0.4.
    beta : float, optional
        Trend smoothing factor between 0 and 1, by default 0.4.
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
    forecast, a, b = np.full((3, historical_length + extra_periods), np.nan)
    # Initialize the first forecast value
    a[0] = demand[0]
    b[0] = demand[1] - demand[0]

    # Create the t+1 forecast until end of the historical period
    for t in range(1, historical_length):
        forecast[t] = a[t - 1] + b[t - 1]
        a[t] = alpha * demand[t] + (1 - alpha) * (a[t - 1] + b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * b[t - 1]

    # Create the t+1 forecast for the extra periods
    for t in range(historical_length, historical_length + extra_periods):
        forecast[t] = a[t - 1] + b[t - 1]
        a[t] = forecast[t]
        b[t] = b[t - 1]

    # Create a DataFrame with the forecast
    forecast_df = pd.DataFrame.from_dict({
        "Demand": demand,
        "Forecast": forecast,
        "Level": a,
        "Trend": b,
        "Error": forecast - demand
    })
    forecast_df.index.name = "Period"

    return forecast_df


def double_exponential_smoothing_damped(
    demand: NDArray,
    alpha: float = 0.4,
    beta: float = 0.4,
    phi: float = 0.9,
    extra_periods: int = 1
) -> pd.DataFrame:
    """
    Computes a damped double exponential smoothing forecast for demand data.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    alpha : float, optional
        Smoothing factor between 0 and 1, by default 0.4.
    beta : float, optional
        Trend smoothing factor between 0 and 1, by default 0.4.
    phi : float, optional
        Damping factor between 0 and 1, by default 0.8.
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
    forecast, a, b = np.full((3, historical_length + extra_periods), np.nan)
    # Initialize the first forecast value
    a[0] = demand[0]
    b[0] = demand[1] - demand[0]

    # Create the t+1 forecast until end of the historical period
    for t in range(1, historical_length):
        forecast[t] = a[t - 1] + phi * b[t - 1]
        a[t] = alpha * demand[t] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]

    # Create the t+1 forecast for the extra periods
    for t in range(historical_length, historical_length + extra_periods):
        forecast[t] = a[t - 1] + phi * b[t - 1]
        a[t] = forecast[t]
        b[t] = phi * b[t - 1]

    # Create a DataFrame with the forecast
    forecast_df = pd.DataFrame.from_dict({
        "Demand": demand,
        "Forecast": forecast,
        "Level": a,
        "Trend": b,
        "Error": forecast - demand
    })
    forecast_df.index.name = "Period"

    return forecast_df
