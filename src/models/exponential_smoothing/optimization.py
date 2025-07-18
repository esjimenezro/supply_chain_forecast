from numpy.typing import NDArray
import numpy as np
import pandas as pd

from models.exponential_smoothing.simple import simple_exponential_smoothing
from models.exponential_smoothing.double import double_exponential_smoothing
from models.exponential_smoothing.double import double_exponential_smoothing_damped


def exponential_smoothing_optimization(
    demand: NDArray,
    extra_periods: int = 6,
) -> pd.DataFrame:
    """
    Optimizes the parameters for simple and double exponential smoothing.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    extra_periods : int, optional
        Number of periods to forecast beyond the historical data, by default 6.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the forecasts from both methods and their errors.
    """
    # Initialization
    params = []  # Contains all the different parameter sets
    kpis = []  # Contains the KPIs for each parameter set
    dfs = []  # Contains the DataFrames returned by the models

    # Loop through different alpha and beta values
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        df = simple_exponential_smoothing(
            demand=demand,
            alpha=alpha,
            extra_periods=extra_periods
        )
        params.append({"Model": "Simple Exponential Smoothing", "Alpha": alpha})
        mae = df["Error"].abs().mean()  # Mean Absolute Error as KPI
        kpis.append(mae)
        dfs.append(df)

        for beta in [0.05, 0.1, 0.2, 0.3, 0.4]:
            df = double_exponential_smoothing(
                demand=demand,
                alpha=alpha,
                beta=beta,
                extra_periods=extra_periods
            )
            params.append({
                "Model": "Double Exponential Smoothing",
                "Alpha": alpha,
                "Beta": beta
            })
            mae = df["Error"].abs().mean()  # Mean Absolute Error as KPI
            kpis.append(mae)
            dfs.append(df)
            for phi in [0.8, 0.85, 0.9, 0.95]:
                df = double_exponential_smoothing_damped(
                    demand=demand,
                    alpha=alpha,
                    beta=beta,
                    phi=phi,
                    extra_periods=extra_periods
                )
                params.append({
                    "Model": "Damped Double Exponential Smoothing",
                    "Alpha": alpha,
                    "Beta": beta,
                    "Phi": phi
                })
                mae = df["Error"].abs().mean()  # Mean Absolute Error as KPI
                kpis.append(mae)
                dfs.append(df)

    best_index = np.argmin(kpis)
    best_params = params[best_index]
    best_kpi = kpis[best_index]
    best_df = dfs[best_index]
    print(f"Best Model: {best_params}, KPI: {best_kpi}")

    return best_df
