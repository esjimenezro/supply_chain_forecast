from numpy.typing import NDArray
import numpy as np

from scipy.stats import norm


def winsorize(
    demand: NDArray,
    lower_percentile: int = 1,
    upper_percentile: int = 99
) -> NDArray:
    """
    Winsorizes the demand data by replacing outliers with the specified
    percentiles.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    lower_percentile : int, optional
        Lower percentile to use for winsorization, by default 1.
    upper_percentile : int, optional
        Upper percentile to use for winsorization, by default 99.

    Returns
    -------
    NDArray
        Winsorized demand data.
    """
    lower_bound = np.percentile(demand, lower_percentile)
    upper_bound = np.percentile(demand, upper_percentile)
    # Replace outliers with the lower and upper bounds
    return np.clip(a=demand, a_min=lower_bound, a_max=upper_bound)


def z_score_smoothing(
    demand: NDArray,
    lower_percentile: int = 1,
    upper_percentile: int = 99
) -> NDArray:
    """
    Replace outliers in the demand data using Z-score method.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    threshold : float, optional
        Z-score threshold to identify outliers, by default 3.0.

    Returns
    -------
    NDArray
        Cleaned demand data.
    """
    mean = np.mean(demand)
    std_dev = np.std(demand)
    lower_bound = norm.ppf(lower_percentile / 100, loc=mean, scale=std_dev)
    upper_bound = norm.ppf(upper_percentile / 100, loc=mean, scale=std_dev)
    # Replace outliers with the threshold
    return np.clip(a=demand, a_min=lower_bound, a_max=upper_bound)


def error_standad_deviation_smoothing_simple(
    demand: NDArray,
    forecast: NDArray,
    lower_percentile: int = 1,
    upper_percentile: int = 99
) -> NDArray:
    """
    Replace outliers in the demand data using error standard deviation method.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    forecast : NDArray
        Forecasted values for the demand.
    lower_percentile : int, optional
        Lower percentile to use for winsorization, by default 1.
    upper_percentile : int, optional
        Upper percentile to use for winsorization, by default 99.

    Returns
    -------
    NDArray
        Cleaned demand data.
    """
    error = demand - forecast
    mean_error = np.mean(error)
    std_dev_error = np.std(error)
    lower_bound = norm.ppf(
        lower_percentile / 100,
        loc=mean_error,
        scale=std_dev_error
    ) + forecast
    upper_bound = norm.ppf(
        upper_percentile / 100,
        loc=mean_error,
        scale=std_dev_error
    ) + forecast
    # Replace outliers with the threshold
    return np.clip(a=demand, a_min=lower_bound, a_max=upper_bound)


def error_standad_deviation_smoothing_compounded(
    demand: NDArray,
    forecast: NDArray,
    lower_percentile: int = 1,
    upper_percentile: int = 99
) -> NDArray:
    """
    Replace outliers in the demand data using error standard deviation method.

    Parameters
    ----------
    demand : NDArray
        Historical demand data as a NumPy array.
    forecast : NDArray
        Forecasted values for the demand.
    lower_percentile : int, optional
        Lower percentile to use for winsorization, by default 1.
    upper_percentile : int, optional
        Upper percentile to use for winsorization, by default 99.

    Returns
    -------
    NDArray
        Cleaned demand data.
    """
    error = demand - forecast
    mean_error = np.mean(error)
    std_dev_error = np.std(error)
    probability = norm.cdf(
        x=error,
        loc=mean_error,
        scale=std_dev_error
    )
    outlier_mask = (
        (probability < lower_percentile / 100)
        | (probability > upper_percentile / 100)
    )
    # Calculate statistics without outliers
    mean_error = np.mean(error[~outlier_mask])
    std_dev_error = np.std(error[~outlier_mask])
    # Calculate bounds based on the statistics
    lower_bound = norm.ppf(
        lower_percentile / 100,
        loc=mean_error,
        scale=std_dev_error
    ) + forecast
    upper_bound = norm.ppf(
        upper_percentile / 100,
        loc=mean_error,
        scale=std_dev_error
    ) + forecast
    # Replace outliers with the threshold
    return np.clip(a=demand, a_min=lower_bound, a_max=upper_bound)
