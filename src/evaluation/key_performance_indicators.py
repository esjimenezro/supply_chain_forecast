from sklearn.base import RegressorMixin

from numpy.typing import NDArray
import pandas as pd
import numpy as np


def kpi(forecast_df: pd.DataFrame) -> None:
    """
    Calculate Key Performance Indicators (KPIs) for the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to evaluate.

    Returns:
    pd.DataFrame: DataFrame containing the KPIs.
    """
    demand_average = forecast_df.loc[
        ~forecast_df["Error"].isna(),
        "Demand"
    ].mean()
    # Bias
    bias_average = forecast_df["Error"].mean()
    bias_relative = bias_average / demand_average
    print("Bias: {:0.2f}, {:.2%}".format(bias_average, bias_relative))
    # MAPE
    mape = (forecast_df["Error"].abs() / forecast_df["Demand"]).mean()
    print("MAPE: {:.2%}".format(mape))
    # MAE
    mae = forecast_df["Error"].abs().mean()
    mae_relative = mae / demand_average
    print("MAE: {:0.2f}, {:.2%}".format(mae, mae_relative))
    # RMSE
    rmse = (forecast_df["Error"] ** 2).mean() ** 0.5
    rmse_relative = rmse / demand_average
    print("RMSE: {:0.2f}, {:.2%}".format(rmse, rmse_relative))


def kpi_ml(
    y_train: NDArray,
    y_train_pred: NDArray,
    y_test: NDArray,
    y_test_pred: NDArray,
    name: str = ""
) -> None:
    """
    Calculate Key Performance Indicators (KPIs) for machine learning predictions.

    Parameters:
    y_train (NDArray): True values for the training set.
    y_train_pred (NDArray): Predicted values for the training set.
    y_test (NDArray): True values for the test set.
    y_test_pred (NDArray): Predicted values for the test set.
    """
    df = pd.DataFrame(columns=["MAE", "RMSE", "BIAS"], index=["Train", "Test"])
    df.index.name = name
    df.loc["Train", "MAE"] = 100 * np.mean(np.abs(y_train - y_train_pred)) / np.mean(y_train)
    df.loc["Train", "RMSE"] = 100 * np.sqrt(np.mean((y_train - y_train_pred) ** 2)) / np.mean(y_train)
    df.loc["Train", "BIAS"] = 100 * (np.mean(y_train - y_train_pred) / np.mean(y_train))

    df.loc["Test", "MAE"] = 100 * np.mean(np.abs(y_test - y_test_pred)) / np.mean(y_test)
    df.loc["Test", "RMSE"] = 100 * np.sqrt(np.mean((y_test - y_test_pred) ** 2)) / np.mean(y_test)
    df.loc["Test", "BIAS"] = 100 * (np.mean(y_test - y_test_pred) / np.mean(y_test))

    df = df.astype(float).round(1)
    print(df)


def model_mae(model: RegressorMixin, X: NDArray, y: NDArray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) of a model's predictions.

    Parameters:
    model (BaseEstimator): The machine learning model to evaluate.
    X (NDArray): Input features for the model.
    y (NDArray): True values for the target variable.

    Returns:
    float: The Mean Absolute Error of the model's predictions.
    """
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred)) / np.mean(y)
    return mae
