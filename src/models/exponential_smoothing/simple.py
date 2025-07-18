from numpy.typing import NDArray

import numpy as np
import pandas as pd


def simple_exponential_smoothing(
    demand: NDArray,
    alpha: float = 0.4,
    extra_periods: int = 1
) -> pd.DataFrame:
    """
    Computes a simple exponential smoothing forecast for demand data.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    alpha : float, optional
        Smoothing factor between 0 and 1, by default 0.4.
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
    forecast = np.full(historical_length + extra_periods, np.nan)
    # Initialize the first forecast value
    forecast[1] = demand[0]

    # Create the t+1 forecast until end of the historical period
    for t in range(2, historical_length + 1):
        forecast[t] = alpha * demand[t - 1] + (1 - alpha) * forecast[t - 1]

    # Create the t+1 forecast for the extra periods
    forecast[t + 1:] = forecast[t]

    # Create a DataFrame with the forecast
    forecast_df = pd.DataFrame.from_dict({
        "Demand": demand,
        "Forecast": forecast,
        "Error": forecast - demand
    })
    forecast_df.index.name = "Period"

    return forecast_df


if __name__ == "__main__":
    demand = np.array([
        28, 19, 18, 13, 19, 16, 19, 18, 13, 16,
        16, 11, 18, 15, 13, 15, 13, 11, 13, 10,
        12
    ])
    moving_average_forecast_df = simple_exponential_smothing(
        demand=demand,
        alpha=0.4,
        extra_periods=4
    )
    moving_average_forecast_df[["Demand", "Forecast"]].plot(
        figsize=(8, 3), title="Simple smoothing", ylim=(0, 30)
    )
