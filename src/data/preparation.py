from numpy.typing import NDArray
import numpy as np
import pandas as pd


def generate_datasets(
    df: pd.DataFrame,
    x_length: int = 12,
    y_length: int = 1,
    test_loops: int = 12
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Generate datasets for time series forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    x_length : int, optional
        Length of the input sequence, by default 12.
    y_length : int, optional
        Length of the output sequence, by default 1.
    test_loops : int, optional
        Number of test loops, by default 12.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the generated datasets.
    """
    data = df.values
    rows, periods = data.shape

    # Training set creation
    loops = periods + 1 - x_length - y_length
    train = []
    for col in range(loops):
        train.append(
            data[:, col:col + x_length + y_length]
        )
    train = np.vstack(train)
    X_train, y_train = np.split(
        train, [-y_length], axis=1
    )

    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(
            X_train, [-rows * test_loops], axis=0
        )
        y_train, y_test = np.split(
            y_train, [-rows * test_loops], axis=0
        )
    else:
        X_test = data[:, -x_length:]
        y_test = np.full((X_test.shape[0], y_length), np.nan)

    # Formatting required for scikit-learn
    if y_length == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    return X_train, y_train, X_test, y_test


def generate_datasets_with_holdout(
    df: pd.DataFrame,
    x_length: int = 12,
    y_length: int = 1,
    test_loops: int = 12,
    holdout_loops: int = 0
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Generate datasets for time series forecasting with holdout.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    x_length : int, optional
        Length of the input sequence, by default 12.
    y_length : int, optional
        Length of the output sequence, by default 1.
    test_loops : int, optional
        Number of test loops, by default 12.
    holdout_loops : int, optional
        Number of holdout loops, by default 0.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
        Tuple containing training and testing datasets.
    """
    data = df.values
    rows, periods = data.shape

    # Training set creation
    train_loops = periods + 1 - x_length - y_length - test_loops
    train = []
    for col in range(train_loops):
        train.append(
            data[:, col:col + x_length + y_length]
        )
    train = np.vstack(train)
    X_train, y_train = np.split(
        train, [-y_length], axis=1
    )

    # Holdout set creation
    if holdout_loops > 0:
        X_train, X_holdout = np.split(
            X_train, [-rows * holdout_loops], axis=0
        )
        y_train, y_holdout = np.split(
            y_train, [-rows * holdout_loops], axis=0
        )
    else:
        X_holdout, y_holdout = np.array([]), np.array([])

    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(
            X_train, [-rows * test_loops], axis=0
        )
        y_train, y_test = np.split(
            y_train, [-rows * test_loops], axis=0
        )
    else:
        X_test = data[:, -x_length:]
        y_test = np.full((X_test.shape[0], y_length), np.nan)

    # Formatting required for scikit-learn
    if y_length == 1:
        y_train = y_train.ravel()
        y_holdout = y_holdout.ravel()
        y_test = y_test.ravel()

    return X_train, y_train, X_holdout, y_holdout, X_test, y_test


def generate_datasets_with_exogenous(
    df: pd.DataFrame,
    X_exogenous: list,
    x_length: int = 12,
    y_length: int = 1,
    test_loops: int = 12
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Generate datasets for time series forecasting with exogenous variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    X_exogenous : list
        List of exogenous variables.
    x_length : int, optional
        Length of the input sequence, by default 12.
    y_length : int, optional
        Length of the output sequence, by default 1.
    test_loops : int, optional
        Number of test loops, by default 12.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        Tuple containing training and testing datasets.
    """
    data = df.values
    rows, periods = data.shape
    X_exo = np.repeat(
        np.reshape(X_exogenous, (1, -1)),
        rows, axis=0
    )
    X_months = np.repeat(
        np.reshape([int(col[-2:]) for col in df.columns], (1, -1)),
        rows,
        axis=0
    )

    # Training set creation
    loops = periods + 1 - x_length - y_length
    train = []
    for col in range(loops):
        months = X_months[:, col + x_length].reshape(-1, 1)
        exo = X_exo[:, col: col + x_length]
        train.append(
            np.hstack(
                (months, exo, data[:, col:col + x_length + y_length])
            )
        )
    train = np.vstack(train)
    X_train, y_train = np.split(
        train, [-y_length], axis=1
    )

    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(
            X_train, [-rows * test_loops], axis=0
        )
        y_train, y_test = np.split(
            y_train, [-rows * test_loops], axis=0
        )
    else:
        X_test = np.hstack(
            (
                months[:, -1].reshape(-1, 1),
                X_exo[:, -x_length:],
                data[:, -x_length:]
            )
        )
        y_test = np.full((X_test.shape[0], y_length), np.nan)

    # Formatting required for scikit-learn
    if y_length == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    return X_train, y_train, X_test, y_test


def generate_datasets_cat(
    df: pd.DataFrame,
    x_length: int = 12,
    y_length: int = 1,
    test_loops: int = 12,
    cat_name: str = "_"
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Generate datasets for time series forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    x_length : int, optional
        Length of the input sequence, by default 12.
    y_length : int, optional
        Length of the output sequence, by default 1.
    test_loops : int, optional
        Number of test loops, by default 12.
    cat_name : str, optional
        Name of the category for the features, by default "_".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the generated datasets.
    """
    cat_cols = [
        col for col in df.columns if cat_name in col
    ]
    data = df.drop(columns=cat_cols).values
    cat_data = df[cat_cols].values
    rows, periods = data.shape

    # Training set creation
    loops = periods + 1 - x_length - y_length
    train = []
    for col in range(loops):
        train.append(
            data[:, col:col + x_length + y_length]
        )
    train = np.vstack(train)
    X_train, y_train = np.split(
        train, [-y_length], axis=1
    )
    X_train = np.hstack(
        (np.vstack([cat_data] * loops), X_train)
    )

    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(
            X_train, [-rows * test_loops], axis=0
        )
        y_train, y_test = np.split(
            y_train, [-rows * test_loops], axis=0
        )
    else:
        X_test = np.hstack((cat_data, data[:, -x_length:]))
        y_test = np.full((X_test.shape[0], y_length), np.nan)

    # Formatting required for scikit-learn
    if y_length == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    return X_train, y_train, X_test, y_test


def generate_datasets_full(
    df: pd.DataFrame,
    X_exogenous: list,
    x_length: int = 12,
    y_length: int = 1,
    test_loops: int = 12,
    holdout_loops: int = 0,
    cat_name: list[str] = ["_"]
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, list[str]]:
    """
    Generate datasets for time series forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    X_exogenous : list
        List of exogenous variables.
    x_length : int, optional
        Length of the input sequence, by default 12.
    y_length : int, optional
        Length of the output sequence, by default 1.
    test_loops : int, optional
        Number of test loops, by default 12.
    holdout_loops : int, optional
        Number of holdout loops, by default 0.
    cat_name : list[str], optional
        Names of the category for the features, by default "_".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the generated datasets.
    """
    cat_cols = [
        col for col in df.columns if any(name in col for name in cat_name)
    ]
    data = df.drop(columns=cat_cols).values
    cat_data = df[cat_cols].values
    rows, periods = data.shape
    X_exo = np.repeat(
        np.reshape(X_exogenous, (1, -1)),
        rows, axis=0
    )
    X_months = np.repeat(
        np.reshape(
            [
                int(col[-2:]) for col in df.columns if col not in cat_cols
            ],
            (1, -1)
        ),
        rows,
        axis=0
    )

    # Training set creation
    loops = periods + 1 - x_length - y_length
    train = []
    for col in range(loops):
        months = X_months[:, col + x_length].reshape(-1, 1)
        exo = X_exo[:, col: col + x_length + y_length]
        exo = np.hstack(
            [
                np.mean(exo, axis=1, keepdims=True),
                np.mean(exo[:, -4:], axis=1, keepdims=True),
                exo
            ]
        )
        data_ = data[:, col:col + x_length + y_length]
        data_augmented = np.hstack(
            [
                np.mean(data_[:, :-y_length], axis=1, keepdims=True),
                np.median(data_[:, :-y_length], axis=1, keepdims=True),
                np.mean(data_[:, -4 - y_length:-y_length], axis=1, keepdims=True),
                np.max(data_[:, :-y_length], axis=1, keepdims=True),
                np.min(data_[:, :-y_length], axis=1, keepdims=True),
                data_
            ]
        )
        train.append(
            np.hstack(
                (months, exo, data_augmented)
            )
        )
    train = np.vstack(train)
    X_train, y_train = np.split(
        train, [-y_length], axis=1
    )
    X_train = np.hstack(
        (np.vstack([cat_data] * loops), X_train)
    )
    features = (
        cat_cols
        + ["Month"]
        + ["Exo Mean", "Exo MA4"]
        + [f"Exo M{-x_length + col}" for col in range(x_length + y_length)]
        + ["Demand Mean", "Demand Median", "Demand MA4", "Demand Max", "Demand Min"]
        + [f"Demand M-{x_length - col}" for col in range(x_length)]
    )

    # Holdout set creation
    if holdout_loops > 0:
        X_train, X_holdout = np.split(
            X_train, [-rows * holdout_loops], axis=0
        )
        y_train, y_holdout = np.split(
            y_train, [-rows * holdout_loops], axis=0
        )
    else:
        X_holdout, y_holdout = np.array([]), np.array([])

    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(
            X_train, [-rows * test_loops], axis=0
        )
        y_train, y_test = np.split(
            y_train, [-rows * test_loops], axis=0
        )
    else:
        exo = X_exo[:, -x_length - y_length:]
        data_ = data[:, -x_length:]
        X_test = np.hstack((
            cat_data,
            months[:, -1].reshape(-1, 1),
            np.hstack(
                [
                    np.mean(exo, axis=1, keepdims=True),
                    np.mean(exo[:, -4:], axis=1, keepdims=True),
                    exo
                ]
            ),
            np.hstack(
                [
                    np.mean(data_, axis=1, keepdims=True),
                    np.median(data_, axis=1, keepdims=True),
                    np.mean(data_[:, -4:], axis=1, keepdims=True),
                    np.max(data_, axis=1, keepdims=True),
                    np.min(data_, axis=1, keepdims=True),
                    data_
                ]
            )
        ))
        y_test = np.full((X_test.shape[0], y_length), np.nan)

    # Formatting required for scikit-learn
    if y_length == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        y_holdout = y_holdout.ravel()

    return X_train, y_train, X_holdout, y_holdout, X_test, y_test, features
