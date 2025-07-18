from data.ingestion import load_car_sales_data
from data.preparation import generate_datasets

from evaluation.key_performance_indicators import kpi_ml
from evaluation.key_performance_indicators import model_mae

from xgboost.sklearn import XGBRegressor
import xgboost as xgb

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == "__main__":
    # Load car sales data
    car_sales_df = load_car_sales_data("../data/norway_new_car_sales_by_make.csv")
    X_train, y_train, X_test, y_test = generate_datasets(
        df=car_sales_df,
        x_length=12,
        y_length=1,
        test_loops=12
    )
    xgb_reg = XGBRegressor(
        n_jobs=-1,
        max_depth=10,
        n_estimators=100,
        learning_rate=0.2,
    )
    xgb_reg.fit(X=X_train, y=y_train)
    xgb_reg.get_booster().feature_names = [f"M{x - 12}" for x in range(12)]
    xgb.plot_importance(
        xgb_reg,
        importance_type="total_gain",
        show_values=False,
    )
    # Multiple output regression
    # X_train, y_train, X_test, y_test = generate_datasets(
    #     df=car_sales_df,
    #     x_length=12,
    #     y_length=6,
    #     test_loops=12
    # )
    # xgb_reg = XGBRegressor(
    #     n_jobs=1,
    #     max_depth=10,
    #     n_estimators=100,
    #     learning_rate=0.2,
    # )
    # multi = MultiOutputRegressor(estimator=xgb_reg, n_jobs=-1)
    # multi.fit(X=X_train, y=y_train)
    # # Future forecast
    # X_train, y_train, X_test, y_test = generate_datasets(
    #     df=car_sales_df,
    #     x_length=12,
    #     y_length=6,
    #     test_loops=0
    # )
    # multi.fit(X=X_train, y=y_train)
    # forecast_df = pd.DataFrame(
    #     data=multi.predict(X=X_test),
    #     index=car_sales_df.index
    # )
    # print(forecast_df)

    # Early stopping
    # x_train, x_val, y_train, y_val = train_test_split(
    #     X_train,
    #     y_train,
    #     test_size=0.15,
    # )

    # xgb_reg = XGBRegressor(
    #     n_jobs=-1,
    #     max_depth=10,
    #     n_estimators=1000,
    #     learning_rate=0.01,
    #     early_stopping_rounds=100,
    #     eval_metric="mae",
    # )
    # xgb_reg.fit(
    #     X=x_train,
    #     y=y_train,
    #     eval_set=[(x_val, y_val)],
    #     verbose=True,
    # )

    # Hyperparameter optimization for XGBRegressor
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train,
    #     y_train,
    #     test_size=0.15,
    # )
    # param_dist = {
    #     "max_depth": [5, 6, 7, 8, 10, 11],
    #     "learning_rate": [0.005, 0.01, 0.025, 0.05, 0.1, 0.15],
    #     "colsample_bynode": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #     "colsample_bylevel": [0.8, 0.9, 1.0],
    #     "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    #     "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #     "min_child_weight": [5, 10, 15, 20, 25],
    #     "reg_alpha": [1, 5, 10, 20, 50],
    #     "reg_lambda": [0.01, 0.05, 0.1, 0.5, 1.0],
    #     "n_estimators": [100],
    # }
    # xgb_reg = XGBRegressor(
    #     n_jobs=1,
    #     early_stopping_rounds=25,
    #     eval_metric="mae",
    # )
    # xgb_cv = RandomizedSearchCV(
    #     estimator=xgb_reg,
    #     param_distributions=param_dist,
    #     n_jobs=-1,
    #     cv=TimeSeriesSplit(n_splits=5),
    #     n_iter=1000,
    #     scoring="neg_mean_absolute_error",
    # )
    # xgb_cv.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)])
    # print(f"Best parameters: {xgb_cv.best_params_}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
    )
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
        early_stopping_rounds=25,
        eval_metric="mae",
        **best_params
    )
    xgb_reg.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)])
    y_train_pred = xgb_reg.predict(X=X_train)
    y_test_pred = xgb_reg.predict(X=X_test)

    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="XGBoost Regressor",
    )
