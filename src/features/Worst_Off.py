import os

from typing import Literal

import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def cleaner_worst_off(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    This function does basic 3 cleaning stuff:
    - 1: cleans the name of columns
    - 2: convert the categorical type
    - 3: round the prices to 4 digits to reduce noise

    At the end it also saves this cleaned data into a csv file in /data/02_preprocessed/ folder.

    Args:
        dataset (pd.DataFrame): a raw European Vanilla dataset

    Returns:
        pd.DataFrame: cleaned version of the same dataset
        Also saves a csv file in data/02_processed/ folder.
    """

    output_file_address = "data/02_processed/Worst_Off_cleaned_dataset.csv"

    # Rename columns for readability
    new_names = {
        "option_type": "option_type",
        "S1": "stock_price_1",
        "S2": "stock_price_2",
        "K1": "strike_price_1",
        "K2": "strike_price_2",
        "sigma1": "volatility_1",
        "sigma2": "volatility_2",
        "q1": "dividend_yield_1",
        "q2": "dividend_yield_2",
        "corr": "correlation",
        "T": "time_to_maturity",
        "r": "interest_rate",
        "price": "price",
    }

    dataset.rename(columns=new_names, inplace=True)

    # Convert categorical type
    dataset["option_type"] = dataset["option_type"].astype("category")

    # Round prices to 4 decimal places
    dataset = dataset.round(4)

    # Save cleaned data to a new CSV file
    os.makedirs(os.path.dirname(output_file_address), exist_ok=True)
    if not os.path.exists(output_file_address):
        dataset.to_csv(output_file_address, index=False)
        print("Cleaned data saved to CSV file.")
    else:
        print("Cleaned data already exists. No new file created.")

    return dataset


def feature_engineering_worst_off(dataset: pd.DataFrame) -> None:
    """
    This function create and apply a pipeline to add new features and scales the data.
    The list is very long, please check out the "add_columns()" function inside this function. (I did it inside here to avoid excessive function definitions)
    It first adds some columns, then add ploynomial features of degree 2 and then scale the numerical variables.
    Then it applies a one_hot encoding on the categorical features and saves the whole dataset as a file in data/02_preprocessed/.

    Parameters:
        dataset (pd.DataFrame): a cleaned european vanilla dataset

    Returns:
        None: It saves the resulting dataset as a file in data/02_preprocessed/
    """

    output_file_address = "data/02_processed/Worst_Off_processed_dataset.csv"

    # Define the features and target variable
    Numerical_features = [
        "stock_price_1",
        "stock_price_2",
        "strike_price_1",
        "strike_price_2",
        "volatility_1",
        "volatility_2",
        "dividend_yield_1",
        "dividend_yield_2",
        "correlation",
        "time_to_maturity",
        "interest_rate",
    ]

    Categorical_features = ["option_type"]

    Target_column = "price"

    # Making a pipeline for preprocessing

    def add_new_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add new columns to the DataFrame as new columns.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing the original features.
            Note: since we are using this function with our own generated dataset, we are already sure that the required columns are present.
            However, to follow the standards and best practices, I am going to add the checker in the beginning of the function.

        Returns:
            pd.DataFrame: The same DataFrame with new features added.
        """

        # Check if the required columns are present in the DataFrame
        required_columns = [
            "stock_price_1",
            "stock_price_2",
            "strike_price_1",
            "strike_price_2",
            "volatility_1",
            "volatility_2",
            "dividend_yield_1",
            "dividend_yield_2",
            "correlation",
            "time_to_maturity",
            "interest_rate",
        ]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"The DataFrame must contain the '{col}' column.")

        # Add new features
        if not 0 in data["time_to_maturity"].values:
            data["1_over_T"] = 1 / data["time_to_maturity"]
            data["log_T"] = np.log(data["time_to_maturity"])

        data["sqrt_T"] = np.sqrt(data["time_to_maturity"])
        data["log1p_T"] = np.log1p(data["time_to_maturity"])

        data["variance_1"] = data["volatility_1"] ** 2
        data["variance_2"] = data["volatility_2"] ** 2

        if not 0 in data["strike_price_1"].values:
            data["stock_1_over_strike_1"] = (
                data["stock_price_1"] / data["strike_price_1"]
            )

        if not 0 in data["strike_price_2"].values:
            data["stock_2_over_strike_2"] = (
                data["stock_price_2"] / data["strike_price_2"]
            )

        if not 0 in data["stock_price_1"].values:
            data["strike_1_over_stock_1"] = (
                data["strike_price_1"] / data["stock_price_1"]
            )

        if not 0 in data["stock_price_2"].values:
            data["strike_2_over_stock_2"] = (
                data["strike_price_2"] / data["stock_price_2"]
            )

        if not 0 in data["interest_rate"].values:
            data["volatility_1_over_interest"] = (
                data["volatility_1"] / data["interest_rate"]
            )
            data["volatility_2_over_interest"] = (
                data["volatility_2"] / data["interest_rate"]
            )
            data["dividend_yield_1_over_interest"] = (
                data["dividend_yield_1"] / data["interest_rate"]
            )
            data["dividend_yield_2_over_interest"] = (
                data["dividend_yield_2"] / data["interest_rate"]
            )
            data["stock_price_1_over_interest"] = (
                data["stock_price_1"] / data["interest_rate"]
            )
            data["stock_price_2_over_interest"] = (
                data["stock_price_2"] / data["interest_rate"]
            )

        if not 0 in data["dividend_yield_1"].values:
            data["volatility_1_over_dividend_1"] = (
                data["volatility_1"] / data["dividend_yield_1"]
            )
            data["interest_rate_over_dividend_1"] = (
                data["interest_rate"] / data["dividend_yield_1"]
            )
            data["stock_price_1_over_dividend_1"] = (
                data["stock_price_1"] / data["dividend_yield_1"]
            )

        if not 0 in data["dividend_yield_2"].values:
            data["volatility_2_over_dividend_2"] = (
                data["volatility_2"] / data["dividend_yield_2"]
            )
            data["interest_rate_over_dividend_2"] = (
                data["interest_rate"] / data["dividend_yield_2"]
            )
            data["stock_price_2_over_dividend_2"] = (
                data["stock_price_2"] / data["dividend_yield_2"]
            )

        if not 0 in data["volatility_1"].values:
            data["interest_rate_over_volatility_1"] = (
                data["interest_rate"] / data["volatility_1"]
            )
            data["dividend_yield_1_over_volatility_1"] = (
                data["dividend_yield_1"] / data["volatility_1"]
            )
            data["stock_price_1_over_volatility_1"] = (
                data["stock_price_1"] / data["volatility_1"]
            )

        if not 0 in data["volatility_2"].values:
            data["interest_rate_over_volatility_2"] = (
                data["interest_rate"] / data["volatility_2"]
            )
            data["dividend_yield_2_over_volatility_2"] = (
                data["dividend_yield_2"] / data["volatility_2"]
            )
            data["stock_price_2_over_volatility_2"] = (
                data["stock_price_2"] / data["volatility_2"]
            )

        return data

    # Adding new columns to the DataFrame
    add_new_cols = FunctionTransformer(func=add_new_columns, validate=False)

    # Adding polynomial features
    add_poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    numerical_pipeline = make_pipeline(
        add_new_cols,
        add_poly,
        StandardScaler(),
    )

    categorical_pipeline = Pipeline(
        [
            ("onehot", OneHotEncoder(drop="if_binary")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerical", numerical_pipeline, Numerical_features),
            ("categorical", categorical_pipeline, Categorical_features),
        ]
    )

    df_temp_1 = add_new_cols.fit_transform(dataset[Numerical_features])
    df_temp_2 = pd.DataFrame(
        add_poly.fit_transform(df_temp_1),
        columns=add_poly.get_feature_names_out(df_temp_1.columns),
    )
    df_temp_2.columns = df_temp_2.columns.str.replace(" ", "_times_", regex=False)

    num_feature_names = df_temp_2.columns.tolist()

    df_temp_1 = pd.DataFrame(
        categorical_pipeline.fit_transform(dataset[Categorical_features]).toarray(),
        columns=categorical_pipeline.get_feature_names_out(Categorical_features),
    )
    cat_feature_names = df_temp_1.columns.tolist()

    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    print("Pipeline tested. Output features are:\n", all_feature_names)

    processed_data = preprocessor.fit_transform(
        dataset[Numerical_features + Categorical_features]
    )
    processed_data = pd.DataFrame(
        processed_data, columns=all_feature_names, index=dataset.index
    )
    processed_data[Target_column] = dataset[Target_column]

    os.makedirs(os.path.dirname(output_file_address), exist_ok=True)
    if not os.path.exists(output_file_address):
        processed_data.to_csv(output_file_address)
        print("Processed data saved to CSV file.")
    else:
        print("Processed data already exists. No new file created.")
