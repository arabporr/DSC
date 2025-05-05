import os
import argparse

from typing import Literal

import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split


from src.features.European_Vanilla import (
    cleaner_european_vanilla,
    feature_engineering_european_vanilla,
)
from src.features.Worst_Off import cleaner_worst_off, feature_engineering_worst_off


def train_test_splitter(
    option: Literal["European_Vanilla", "Worst_Off"], train_idx, test_idx
) -> None:
    """
    This function loads the data based on the option data type and then splits it into training and testing splits.
    Finally save them in data/03_splitted/ folder. (for European_Vanilla in data/03_splitted/European_Vanilla/ and
    for the Worst_Off in the data/03_splitted/Worst_Off/ folder)

    Parameters:
        option (str): the type of option data we want to split
        train_idx (pandas.core.indexes.base.Index): training set indexes
        test_idx (pandas.core.indexes.base.Index): testing set indexes

    Returns:
        None: It saves the result as csv files in /data/03_splitted/ folder.
    """
    # Output paths
    output_directory = f"data/03_splitted/{option}/"
    train_file_address = f"data/03_splitted/{option}/training_data.csv"
    test_file_address = f"data/03_splitted/{option}/testing_data.csv"

    # Load and preprocess data based on the option
    if option == "European_Vanilla":
        processed_data = pd.read_csv(
            "data/02_processed/European_Vanilla_processed_dataset.csv"
        )
    elif option == "Worst_Off":
        processed_data = pd.read_csv(
            "data/02_processed/Worst_Off_processed_dataset.csv"
        )
    else:
        raise ValueError(
            "Invalid option. Choose either 'European_Vanilla' or 'Worst_Off'."
        )

    # Split the data into features and target
    if option == "European_Vanilla":
        target_column = "black_scholes_price"
    elif option == "Worst_Off":
        target_column = "price"

    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]

    # now use the train_idx and test_idx you already generated
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    # Save the splits to CSV files
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)

    train_data = pd.concat([X_train, y_train], axis=1)
    if not os.path.exists(train_file_address):
        train_data.to_csv(train_file_address, index=False)
        print(f"Training data saved to {train_file_address}.")
    else:
        print("Training data already exists. No new file created.")

    test_data = pd.concat([X_test, y_test], axis=1)
    if not os.path.exists(test_file_address):
        test_data.to_csv(test_file_address, index=False)
        print(f"Test data saved to {test_file_address}.")
    else:
        print("Test data already exists. No new file created.")


def preprocessor(option: Literal["European_Vanilla", "Worst_Off"]) -> None:
    """
    Wrapper function for cleaning dataset.
    It cleans data first, then add new features and finally splits it into training and test sets.
    Along this process, it also saves the results as csv files in the data folder.

    Parameters:
        option (str): the type of option data

    Returns:
        None: it saves the preprocessed data into csv files in data/02_preprocessed/ and /data/03_splitted/ folders.
    """
    European_Vanilla_raw_address = "data/01_raw/European_Vanilla_dataset.csv"
    Worst_Off_raw_address = "data/01_raw/Worst_Off_dataset.csv"

    if option == "European_Vanilla":
        raw_data = pd.read_csv(European_Vanilla_raw_address)

        cleaned_data = cleaner_european_vanilla(raw_data)
        train_idx, test_idx = train_test_split(
            raw_data.index, test_size=0.3, random_state=2025, shuffle=True
        )
        feature_engineering_european_vanilla(cleaned_data, train_idx, test_idx)
    elif option == "Worst_Off":
        raw_data = pd.read_csv(Worst_Off_raw_address)
        cleaned_data = cleaner_worst_off(raw_data)
        # Since I tested and the file size was above 15GB! I decided to sample from this dataset to avoid memory issue
        # (it has 780 variables and 1,000,000 rows!) we will use 10% for now, which has 100K rows and is 1.5 GB
        cleaned_data = cleaned_data.sample(frac=0.1, random_state=2025)
        train_idx, test_idx = train_test_split(
            raw_data.index, test_size=0.3, random_state=2025, shuffle=True
        )
        feature_engineering_worst_off(cleaned_data, train_idx, test_idx)
    else:
        raise ValueError(
            "Invalid option. Choose either 'European_Vanilla' or 'Worst_Off'."
        )

    # Split the data into training and testing sets
    train_test_splitter(option, train_idx, test_idx)


if __name__ == "__main__":
    """
    Adding the functionality of execution from command line.
    How to use?

    python src/features/preprocess.py {Worst_Off or European_Vanilla}

    It automatically saves the data into data/02_processed/ and data/splitted/ folders.
    """

    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["European_Vanilla", "Worst_Off"])
    args = p.parse_args()
    preprocessor(args.version)
