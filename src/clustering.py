import pandas as pd
from data.ingestion import load_car_sales_data

from sklearn.cluster import KMeans

import calendar

import seaborn as sns


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


if __name__ == "__main__":
    # Load car sales data
    car_sales_df = load_car_sales_data("../data/norway_new_car_sales_by_make.csv")
    # Calculate seasonal factors
    seasonal_df = seasonal_factors(car_sales_df)
    # Scale seasonal factors
    scaled_seasonal_df = scaler(seasonal_df)
    print(scaled_seasonal_df.head())

    # KMeans clustering
    kmeans0 = KMeans(n_clusters=4, random_state=42)
    scaled_seasonal_df["Cluster"] = kmeans0.fit_predict(scaled_seasonal_df)
    print(scaled_seasonal_df.head())

    # Inertia
    results = []
    for n_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_seasonal_df.drop(columns=["Cluster"]))
        results.append([n_clusters, kmeans.inertia_])
    inertia_df = pd.DataFrame(results, columns=["n_clusters", "inertia"]).set_index("n_clusters")
    inertia_df.plot()

    # Cluster visualization
    centers = pd.DataFrame(kmeans0.cluster_centers_).transpose()
    centers.index = calendar.month_abbr[1:]
    centers.columns = [f"Cluster {i}" for i in range(centers.shape[1])]
    sns.heatmap(
        centers,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="RdBu_r"
    )
