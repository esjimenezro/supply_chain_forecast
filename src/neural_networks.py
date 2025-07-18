from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

from data.ingestion import load_car_sales_data
from data.preparation import generate_datasets

from evaluation.key_performance_indicators import kpi_ml


if __name__ == "__main__":
    # Load car sales data
    car_sales_df = load_car_sales_data("../data/norway_new_car_sales_by_make.csv")
    X_train, y_train, X_test, y_test = generate_datasets(
        df=car_sales_df,
        x_length=12,
        y_length=1,
        test_loops=12
    )
    # Define the parameter distribution for MLPRegressor
    hidden_layer_sizes = [
        [neuron] * hidden_layer
        for neuron in range(10, 60, 10)
        for hidden_layer in range(2, 7)
    ]
    alpha = [5, 1, 0.5, 0.1, 0.05, 0.01, 0.001]
    learning_rate_init = [0.05, 0.01, 0.005, 0.001, 0.0005]
    beta_1 = [0.85, 0.875, 0.9, 0.95, 0.975, 0.99, 0.995]
    beta_2 = [0.99, 0.995, 0.999, 0.9995, 0.9999]
    param_distribution = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "beta_1": beta_1,
        "beta_2": beta_2,
    }
    # Fixed parameters
    activation = "relu"
    solver = "adam"
    early_stopping = True
    n_iter_no_change = 50
    validation_fraction = 0.1
    tol = 0.0001
    param_fixed = {
        "activation": activation,
        "solver": solver,
        "early_stopping": early_stopping,
        "n_iter_no_change": n_iter_no_change,
        "validation_fraction": validation_fraction,
        "tol": tol,
    }

    # Create the MLPRegressor model with the defined parameters
    model = MLPRegressor(
        **param_fixed,
        **{'learning_rate_init': 0.01, 'hidden_layer_sizes': [30, 30, 30, 30], 'beta_2': 0.995, 'beta_1': 0.995, 'alpha': 5}
    )
    model.fit(X=X_train, y=y_train)
    # cross_val = RandomizedSearchCV(
    #     estimator=model,
    #     param_distributions=param_distribution,
    #     cv=TimeSeriesSplit(n_splits=5),
    #     verbose=2,
    #     n_jobs=-1,
    #     n_iter=200,
    #     scoring="neg_mean_absolute_error",
    # )
    # cross_val.fit(X=X_train, y=y_train)
    # print(f"Best parameters: {cross_val.best_params_}")
    y_train_pred = model.predict(X=X_train)
    y_test_pred = model.predict(X=X_test)
    kpi_ml(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        name="MLPRegressor",
    )
