from data.ingestion import load_car_sales_data
from data.preparation import generate_datasets_full
from evaluation.key_performance_indicators import kpi_ml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from xgboost.sklearn import XGBRegressor
import calendar


def seasonal_factors(
    df: pd.DataFrame,
    season_length: int = 12
) -> pd.DataFrame:
    """
    Calculate seasonal factors for each period in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    season_length : int, optional
        Length of the seasonal period, by default 12.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the seasonal factors.
    """
    seasonal_df = pd.DataFrame(index=df.index)
    for period in range(season_length):
        seasonal_df[period + 1] = df.iloc[:, period::season_length].mean(axis=1)
    seasonal_df = seasonal_df.divide(
        seasonal_df.mean(axis=1),
        axis=0
    ).fillna(0)
    return seasonal_df


def scaler(seasonal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the seasonal factors to a range of 0 to 1.

    Parameters
    ----------
    seasonal_df : pd.DataFrame
        DataFrame containing the seasonal factors.

    Returns
    -------
    pd.DataFrame
        Scaled DataFrame with seasonal factors.
    """
    mean = seasonal_df.mean(axis=1)
    max_i = seasonal_df.max(axis=1)
    min_i = seasonal_df.min(axis=1)
    seasonal_df = seasonal_df.subtract(mean, axis=0)
    seasonal_df = seasonal_df.divide(
        max_i - min_i,
        axis=0
    ).fillna(0)
    return seasonal_df


def model_kpi(model, X, y):
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred)) / np.mean(y)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2)) / np.mean(y)
    return mae, rmse


if __name__ == "__main__":
    # Load car sales data & GDP data
    car_sales_df = load_car_sales_data("../data/norway_new_car_sales_by_make.csv")
    gdp_df = pd.read_excel("../data/GDP.xlsx").set_index("Year")
    dates = pd.to_datetime(car_sales_df.columns, format="%Y-%m").year
    X_gdp = [gdp_df.loc[date, "GDP"] for date in dates]

    # Define car categories
    luxury = [
        "Aston Martin", "Bentley", "Ferrari", "Lamborghini", "Lexus",
        "Lotus", "Maserati", "McLaren", "Porsche", "Tesla",
    ]
    premium = [
        "Audi", "BMW", "Cadillac", "Infinity", "Land Rover", "MINI",
        "Mercedes-Benz", "Jaguar",
    ]
    low_cost = [
        "Dacia", "Skoda"
    ]
    # Define segments
    car_sales_df["Segment"] = 2
    mask = car_sales_df.index.isin(luxury)
    car_sales_df.loc[mask, "Segment"] = 4  # Luxury
    mask = car_sales_df.index.isin(premium)
    car_sales_df.loc[mask, "Segment"] = 3  # Premium
    mask = car_sales_df.index.isin(low_cost)
    car_sales_df.loc[mask, "Segment"] = 1  # Low cost

    # One-hot encoding of categorical features
    car_sales_df["Brand"] = car_sales_df.index
    car_sales_df = pd.get_dummies(
        car_sales_df,
        columns=["Brand"],
        prefix_sep="_",
    )
    car_sales_df = car_sales_df.drop(columns=["Brand"], errors="ignore")

    # Calculate seasonal factors
    seasonal_df = seasonal_factors(car_sales_df)
    # Scale seasonal factors
    scaled_seasonal_df = scaler(seasonal_df)
    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    car_sales_df["Group"] = kmeans.fit_predict(scaled_seasonal_df)

    # Generate datasets for forecasting
    X_train, y_train, X_holdout, y_holdout, X_test, y_test, features = generate_datasets_full(
        df=car_sales_df,
        X_exogenous=X_gdp,
        x_length=12,
        y_length=1,
        test_loops=12,
        holdout_loops=0,
        cat_name=["Brand_", "Segment", "Group"]
    )
    # Validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    # Train XGBoost regressor
    best_params = {
        'subsample': 0.2,
        'reg_lambda': 0.1,
        'reg_alpha': 20,
        'n_estimators': 1000,
        'min_child_weight': 5,
        'max_depth': 10,
        'learning_rate': 0.005,
        'colsample_bytree': 0.8,
        'colsample_bynode': 1.0,
        'colsample_bylevel': 0.9
    }
    xgb_reg = XGBRegressor(
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric="mae",
        **best_params
    )
    xgb_reg.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    # Predict on train and test sets
    y_train_pred = xgb_reg.predict(X_train)
    y_test_pred = xgb_reg.predict(X_test)
    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="XGBRegressor with Seasonal Factors",
    )

    # Feature importance
    importance = xgb_reg.get_booster().get_score(importance_type="total_gain")
    importance_df = pd.DataFrame.from_dict(
        importance, orient="index", columns=["Importance"]
    )
    importance_df.index = np.array(features)[importance_df.index.astype(str).str.replace("f", "").astype(int)]
    importance_df = (importance_df["Importance"] / importance_df["Importance"].sum()).sort_values(ascending=False)
    print(importance_df.head(10))

    # Backward elimination
    results = []
    limits = [
        0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004,
        0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.002, 0.004,
        0.008, 0.01, 0.02, 0.04, 0.06
    ]
    for limit in limits:
        mask = [feature in importance_df[importance_df > limit] for feature in features]
        xgb_reg.fit(
            X=X_train[:, mask],
            y=y_train,
            eval_set=[(X_val[:, mask], y_val)],
            verbose=False
        )
        results.append(
            model_kpi(
                model=xgb_reg,
                X=X_val[:, mask],
                y=y_val
            )
        )
    results_df = pd.DataFrame(
        data=results,
        index=limits,
        columns=["MAE", "RMSE"]
    )
    results_df.plot(
        secondary_y="MAE",
        logx=True,
    )

    # Run model with best features
    mask = [feature in importance_df[importance_df > 0.007] for feature in features]
    xgb_reg.fit(
        X=X_train[:, mask],
        y=y_train,
        eval_set=[(X_val[:, mask], y_val)],
        verbose=True
    )
    y_train_pred = xgb_reg.predict(X_train[:, mask])
    y_test_pred = xgb_reg.predict(X_test[:, mask])
    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="XGBRegressor with Best Features",
    )
