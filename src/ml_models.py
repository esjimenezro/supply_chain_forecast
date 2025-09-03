from data.ingestion import load_car_sales_data
from data.preparation import generate_datasets
from evaluation.key_performance_indicators import kpi_ml

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
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
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    


    # Feature importance for Random Forest Regressor
    forest: RandomForestRegressor = random_search.best_estimator_
    cols = X_train.shape[1]
    features = [f"M-{cols - col}" for col in range(cols)]
    data = forest.feature_importances_.reshape(-1, 1)
    feature_importance_df = pd.DataFrame(
        data=data,
        index=features,
        columns=["Forest"]
    )
    feature_importance_df.plot(kind="bar")

    # Extra Trees Regressor model
    extra_trees = ExtraTreesRegressor(
        n_jobs=-1,
        n_estimators=200,
        min_samples_split=15,
        min_samples_leaf=4,
        max_samples=0.95,
        max_features=4,
        max_depth=8,
        bootstrap=True,
    )
    extra_trees.fit(X=X_train, y=y_train)
    y_train_pred = extra_trees.predict(X=X_train)
    y_test_pred = extra_trees.predict(X=X_test)

    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="Extra Trees Regressor"
    )

    # Hyperparameter optimization for Extra Trees Regressor
    max_depth = list(range(6, 13)) + [None]
    min_samples_split = list(range(7, 16))
    min_samples_leaf = list(range(2, 13))
    max_features = list(range(5, 13))
    bootstrap = [True]
    max_samples = [0.7, 0.8, 0.9, 0.95, 1.0]
    param_dist = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "max_samples": max_samples,
    }

    extra_trees = ExtraTreesRegressor(n_jobs=1, n_estimators=200)
    random_search = RandomizedSearchCV(
        estimator=extra_trees,
        param_distributions=param_dist,
        n_jobs=-1,
        cv=TimeSeriesSplit(n_splits=5),
        verbose=1,
        n_iter=400,  # 400
        scoring="neg_mean_absolute_error",
    )
    random_search.fit(X=X_train, y=y_train)
    print("Best parameters found: ", random_search.best_params_)

    y_train_pred = random_search.predict(X=X_train)
    y_test_pred = random_search.predict(X=X_test)
    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="Extra Trees Regressor with Random Search"
    )

    # Predicting future values
    X_train, y_train, X_test, y_test = generate_datasets(
        df=car_sales_df,
        x_length=12,
        y_length=1,
        test_loops=0
    )
    reg.fit(X=X_train, y=y_train)
    forecast = pd.DataFrame(
        data=reg.predict(X=X_test),
        index=car_sales_df.index,
        columns=["Forecast"]
    )
    print(forecast)