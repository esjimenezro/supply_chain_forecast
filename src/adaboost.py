from data.ingestion import load_car_sales_data
from data.preparation import generate_datasets

from evaluation.key_performance_indicators import kpi_ml
from evaluation.key_performance_indicators import model_mae

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
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
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=8),
        n_estimators=100,
        learning_rate=0.25,
        loss="square"
    )
    ada.fit(X=X_train, y=y_train)
    y_train_pred = ada.predict(X=X_train)
    y_test_pred = ada.predict(X=X_test)
    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="AdaBoost Regressor"
    )
    # Hyperparameter optimization for AdaBoost Regressor
    # learning_rate = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # loss = ["linear", "square", "exponential"]
    # param_dist = {
    #     "learning_rate": learning_rate,
    #     "loss": loss
    # }
    # results = []
    # for max_depth in range(2, 18, 2):
    #     ada = AdaBoostRegressor(
    #         estimator=DecisionTreeRegressor(max_depth=max_depth),
    #         n_estimators=100,
    #     )
    #     ada_cv = RandomizedSearchCV(
    #         estimator=ada,
    #         param_distributions=param_dist,
    #         n_jobs=-1,
    #         cv=TimeSeriesSplit(n_splits=5),
    #         n_iter=20,
    #         scoring="neg_mean_absolute_error",
    #         verbose=1,
    #     )
    #     ada_cv.fit(X=X_train, y=y_train)
    #     print(f"Best parameters for max_depth={max_depth}: {ada_cv.best_params_}")
    #     print(f"Best score for max_depth={max_depth}: {ada_cv.best_score_}")
    #     results.append([
    #         ada_cv.best_score_,
    #         ada_cv.best_params_,
    #         max_depth
    #     ])

    # results_df = pd.DataFrame(
    #     data=results,
    #     columns=["best_score", "best_params", "max_depth"]
    # )
    # print(results_df.sort_values(by="best_score", ascending=False))
    # Optimized AdaBoost Regressor
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=8),
        n_estimators=100,
        learning_rate=0.005,
        loss="square"
    )
    ada.fit(X=X_train, y=y_train)
    y_train_pred = ada.predict(X=X_train)
    y_test_pred = ada.predict(X=X_test)
    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="Optimized AdaBoost Regressor"
    )
    
