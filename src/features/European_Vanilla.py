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


def cleaner_european_vanilla(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    This function does basic 3 cleaning stuff:
    - 1: cleans the name of columns
    - 2: convert the categorical type
    - 3: round the prices to 4 digits to reduce noise

    At the end it also saves this cleaned data into a csv file in /data/02_preprocessed/ folder.

    Parameters:
        dataset (pd.DataFrame): a raw european Vanilla dataset

    Returns:
        pd.DataFrame: cleaned version of the same dataset
        Also saves a csv file in data/02_processed/ folder.
    """

    output_file_address = "data/02_processed/European_Vanilla_cleaned_dataset.csv"

    # Rename columns for readability
    new_names = {
        "option_type": "option_type",
        "S": "stock_price",
        "K": "strike_price",
        "T": "time_to_maturity",
        "r": "interest_rate",
        "sigma": "volatility",
        "q": "dividend_yield",
        "bs_price": "black_scholes_price",
        "mc_price": "monte_carlo_price",
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


def feature_engineering_european_vanilla(
    dataset: pd.DataFrame, train_idx, test_idx
) -> None:
    """
    This function create and apply a pipeline to add new features and scales the data.
    The list is very long, please check out the "add_columns()" function inside this function. (I did it inside here to avoid excessive function definitions)
    It first adds some columns, then add ploynomial features of degree 2 and then scale the numerical variables.
    Then it applies a one_hot encoding on the categorical features and saves the whole dataset as a file in data/02_preprocessed/.

    Parameters:
        dataset (pd.DataFrame): a cleaned european vanilla dataset
        train_idx (pandas.core.indexes.base.Index): training set indexes
        test_idx (pandas.core.indexes.base.Index): testing set indexes

    Returns:
        None: It saves the resulting dataset as a file in data/02_preprocessed/
    """

    output_file_address = "data/02_processed/European_Vanilla_processed_dataset.csv"

    # Define the features and target variable
    Numerical_features = [
        "stock_price",
        "strike_price",
        "time_to_maturity",
        "interest_rate",
        "volatility",
        "dividend_yield",
    ]

    Categorical_features = ["option_type"]

    Target_column = "black_scholes_price"

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
        if "stock_price" not in data.columns:
            raise ValueError("The DataFrame must contain the 'stock_price' column.")
        if "strike_price" not in data.columns:
            raise ValueError("The DataFrame must contain the 'strike_price' column.")
        if "time_to_maturity" not in data.columns:
            raise ValueError(
                "The DataFrame must contain the 'time_to_maturity' column."
            )
        if "volatility" not in data.columns:
            raise ValueError("The DataFrame must contain the 'volatility' column.")
        if "interest_rate" not in data.columns:
            raise ValueError("The DataFrame must contain the 'interest_rate' column.")
        if "dividend_yield" not in data.columns:
            raise ValueError("The DataFrame must contain the 'dividend_yield' column.")

        if not 0 in data["time_to_maturity"].values:
            data["1_over_T"] = 1 / data["time_to_maturity"]
            data["log_T"] = np.log(data["time_to_maturity"])

        data["sqrt_T"] = np.sqrt(data["time_to_maturity"])
        data["log1p_T"] = np.log1p(data["time_to_maturity"])

        data["variance"] = data["volatility"] ** 2

        if not 0 in data["strike_price"].values:
            data["stock_over_strike"] = data["stock_price"] / data["strike_price"]

        if not 0 in data["stock_price"].values:
            data["strike_over_stock"] = data["strike_price"] / data["stock_price"]

        if not 0 in data["interest_rate"].values:
            data["volatility_over_interest"] = (
                data["volatility"] / data["interest_rate"]
            )
            data["dividend_yield_over_interest"] = (
                data["dividend_yield"] / data["interest_rate"]
            )
            data["stock_price_over_interest"] = (
                data["stock_price"] / data["interest_rate"]
            )

        if not 0 in data["dividend_yield"].values:
            data["volatility_over_dividend"] = (
                data["volatility"] / data["dividend_yield"]
            )
            data["interest_rate_over_dividend"] = (
                data["interest_rate"] / data["dividend_yield"]
            )
            data["stock_price_over_dividend"] = (
                data["stock_price"] / data["dividend_yield"]
            )

        if not 0 in data["volatility"].values:
            data["interest_rate_over_volatility"] = (
                data["interest_rate"] / data["volatility"]
            )
            data["dividend_yield_over_volatility"] = (
                data["dividend_yield"] / data["volatility"]
            )
            data["stock_price_over_volatility"] = (
                data["stock_price"] / data["volatility"]
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

    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    df_temp_1 = add_new_cols.fit_transform(train_df[Numerical_features])
    df_temp_2 = pd.DataFrame(
        add_poly.fit_transform(df_temp_1),
        columns=add_poly.get_feature_names_out(df_temp_1.columns),
    )
    df_temp_2.columns = df_temp_2.columns.str.replace(" ", "_times_", regex=False)

    num_feature_names = df_temp_2.columns.tolist()

    df_temp_1 = pd.DataFrame(
        categorical_pipeline.fit_transform(train_df[Categorical_features]).toarray(),
        columns=categorical_pipeline.get_feature_names_out(Categorical_features),
    )
    cat_feature_names = df_temp_1.columns.tolist()

    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    print("Pipeline tested. Output features are:\n", all_feature_names)

    processed_data_train = preprocessor.fit_transform(
        train_df[Numerical_features + Categorical_features]
    )
    processed_data_test = preprocessor.transform(
        test_df[Numerical_features + Categorical_features]
    )
    processed_data = np.vstack([processed_data_train, processed_data_test])
    processed_data = pd.DataFrame(
        processed_data, columns=all_feature_names, index=dataset.index
    )
    processed_data[Target_column] = pd.concat(
        [train_df[Target_column], test_df[Target_column]], axis=0
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_file_address), exist_ok=True)
    if not os.path.exists(output_file_address):
        processed_data.to_csv(output_file_address, index=False)
        print("Processed data saved to CSV file.")
    else:
        print("Processed data already exists. No new file created.")
