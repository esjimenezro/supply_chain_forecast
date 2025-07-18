import pandas as pd


def load_car_sales_data(
    file_path: str = "data/norway_new_car_sales_by_make.csv"
) -> pd.DataFrame:
    """
    Load car sales data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing car sales data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the car sales data.
    """
    data = pd.read_csv(file_path)
    data["Period"] = (
        data["Year"].astype(str)
        + "-"
        + data["Month"].astype(str).str.zfill(2)
    )
    df = pd.pivot_table(
        data=data,
        values="Quantity",
        index="Make",
        columns="Period",
        aggfunc="sum",
        fill_value=0
    )
    return df


if __name__ == "__main__":
    # Load car sales data
    car_sales_df = load_car_sales_data("../../data/norway_new_car_sales_by_make.csv")
    print(car_sales_df.head())
