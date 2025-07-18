from data.ingestion import load_car_sales_data
from data.preparation import generate_datasets_cat

import pandas as pd


if __name__ == "__main__":
    # Load car sales data
    car_sales_df = load_car_sales_data("../data/norway_new_car_sales_by_make.csv")
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
    X_train, y_train, X_test, y_test = generate_datasets_cat(
        df=car_sales_df,
        x_length=12,
        y_length=1,
        test_loops=12,
        cat_name="Segment"
    )
    # car_sales_df["Brand"] = car_sales_df.index
    # Integer encoding of categorical features
    # car_sales_df["Brand"] = car_sales_df["Brand"].astype("category").cat.codes
    # One-hot encoding of categorical features
    # car_sales_df = pd.get_dummies(
    #     car_sales_df,
    #     columns=["Brand"],
    #     prefix_sep="_",
    # )
    # X_train, y_train, X_test, y_test = generate_datasets_cat(
    #     df=car_sales_df,
    #     x_length=12,
    #     y_length=1,
    #     test_loops=12,
    #     cat_name="_"
    # )
