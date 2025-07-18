import numpy as np

from models.ma.moving_average import moving_average
from models.exponential_smoothing.simple import simple_exponential_smoothing
from models.exponential_smoothing.double import double_exponential_smoothing
from evaluation.key_performance_indicators import kpi
from models.exponential_smoothing.optimization import exponential_smoothing_optimization
from models.exponential_smoothing.triple import triple_exponential_smoothing_mul
from models.exponential_smoothing.triple import triple_exponential_smoothing_add


if __name__ == "__main__":
    # Simple models
    demand = np.array(
        [37, 60, 85, 112, 132, 145, 179, 198, 150, 132]
    )
    forecast_df = moving_average(
        demand=demand,
        window_size=3,
        extra_periods=4
    )
    kpi(forecast_df=forecast_df)
    forecast_df = simple_exponential_smoothing(
        demand=demand,
        alpha=0.4,
        extra_periods=4
    )
    kpi(forecast_df=forecast_df)
    forecast_df = double_exponential_smoothing(
        demand=demand,
        alpha=0.4,
        beta=0.4,
        extra_periods=4
    )
    kpi(forecast_df=forecast_df)
    # Optimization
    demand = np.array(
        [28, 19, 18, 13, 19, 16, 19, 18, 13, 16,
         16, 11, 18, 15, 13, 15, 13, 11, 13, 10, 12]
    )
    forecast_df = exponential_smoothing_optimization(
        demand=demand,
        extra_periods=6
    )
    forecast_df[["Demand", "Forecast"]].plot(
        figsize=(8, 3), title="Simple smoothing", ylim=(0, 30)
    )
    # Triple exponential smoothing
    demand = np.array([
        14, 10, 6, 2, 18, 8, 4, 1, 16, 9,
        5, 3, 18, 11, 4, 2, 17, 9, 5, 1
    ])
    forecast_df = triple_exponential_smoothing_mul(
        demand=demand,
        alpha=0.4,
        beta=0.4,
        phi=0.9,
        gamma=0.3,
        season_length=12,
        extra_periods=4
    )
    kpi(forecast_df=forecast_df)
    forecast_df[["Demand", "Forecast"]].plot(
        figsize=(8, 3), title="Triple Exponential Smoothing", ylim=(0, 25)
    )
    forecast_df[["Level", "Trend", "Season"]].plot(
        figsize=(8, 3), secondary_y=["Season"]
    )
    forecast_df = triple_exponential_smoothing_add(
        demand=demand,
        alpha=0.4,
        beta=0.4,
        phi=0.9,
        gamma=0.3,
        season_length=12,
        extra_periods=4
    )
    kpi(forecast_df=forecast_df)
    forecast_df[["Demand", "Forecast"]].plot(
        figsize=(8, 3), title="Triple Exponential Smoothing", ylim=(0, 25)
    )
    forecast_df[["Level", "Trend", "Season"]].plot(
        figsize=(8, 3), secondary_y=["Season"]
    )
