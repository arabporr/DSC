import os
import argparse
import pickle

from typing import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

from src.models.model_families.baseline import train_baseline_models
from src.models.model_families.tree_based import train_tree_models
from src.models.model_families.kernel_based import train_kernel_models
from src.models.model_families.neural_network import train_nn_models


def train_all_models(option: Literal["European_Vanilla", "Worst_Off"]) -> None:
    """
    This function loads the train and test datasets based on the option data type and then trains different families of models on the train data.
    The training is being done in sci-kit learn style (I avoided using TensorFlow and PyTorch for the neural network due to simplicity of the task and limited time)
    We train each model with a grid of parameters and 3-fold cross-validation.
    At the end, we create a csv and a pickle file containing the results of our models (Root Mean Square Error) on training and testing parts.

    Parameters:
        option (str): the type of option data we want to use

    Returns:
        None: It saves the result as csv files in /results/ folder.
    """
    # Output paths
    if option == "European_Vanilla":
        output_dir = "results/European_Vanilla"
        output_file_address_result = "results/European_Vanilla/model_comparison.csv"
        output_file_address_details = "results/European_Vanilla/model_comparison.csv"
    elif option == "Worst_Off":
        output_dir = "results/Worst_Off"
        output_file_address_result = "results/Worst_Off/model_comparison.csv"
        output_file_address_details = "results/Worst_Off/model_comparison.csv"
    else:
        raise ValueError(
            "Invalid option. Choose either 'European_Vanilla' or 'Worst_Off'."
        )

    # Data loading

    if option == "European_Vanilla":
        training_data = pd.read_csv(
            "data/03_splitted/European_Vanilla/training_data.csv"
        )
        test_data = pd.read_csv("data/03_splitted/European_Vanilla/testing_data.csv")
    else:
        training_data = pd.read_csv("data/03_splitted/Worst_Off/training_data.csv")
        test_data = pd.read_csv("data/03_splitted/Worst_Off/testing_data.csv")

    X_train = training_data.iloc[:, :-1]
    y_train = training_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    print("Model Trainer: ", "Data loaded successfully")

    # Data preparation
    # Identifying columns with almost constant values
    Columns_variance_filter = X_train.loc[:, X_train.var() > 1e-2].columns.tolist()

    print("Model Trainer: ", "Low variance columns identified")

    # Identifying columns with low correlation to the target variable
    df = X_train.copy()
    df["y"] = y_train
    corr = df.corr()
    Columns_correlation_filter = corr["y"].abs()[corr["y"].abs() > 0.05].index.tolist()

    print("Model Trainer: ", "Columns with low correlated with target identified")

    # Identifying top 40 features using mutual information regression
    Feature_selection = SelectKBest(f_regression, k=40).fit(X_train, y_train)
    Top_40_columns = X_train.columns[Feature_selection.get_support(indices=True)]

    print("Model Trainer: ", "Top 40 variables identified")

    # Getting the intersection of the methods above
    top_variables = set(Top_40_columns).intersection(
        set(Columns_variance_filter).intersection(set(Columns_correlation_filter))
    )

    X_train_top_vars = X_train[list(top_variables)]
    X_test_top_vars = X_test[list(top_variables)]

    # Also making a smaller set of features using PCA
    pca_full = PCA(random_state=2025)
    pca_full.fit(X_train)

    print("Model Trainer: ", "PCA for full dataset calculated")

    # Adding the amount of variance explained by each component
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Setting a threshold for the amount of variance to be explained
    # Here I just tried different values since 95 till 99 were in 20-ish range
    # I decided to go with 98%, not too crazy and not too low
    Var_threshold = 0.98
    Number_of_components = (
        np.where(cum_var > Var_threshold)[0][0] + 1
    )  # this the index of the first component that explains more than 98% of the variance

    print("Model Trainer: ", f"{Number_of_components} components selected (98% var)")

    pca = PCA(n_components=Number_of_components, random_state=2025)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # For later use, we will create a dictionary with the different data modes
    # this will be used to train the models and will make our job easier
    # when we want to train different models with different data modes
    data_modes = {
        "all_variables": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        "top_vars": {
            "X_train": X_train_top_vars,
            "X_test": X_test_top_vars,
            "y_train": y_train,
            "y_test": y_test,
        },
        "pca": {
            "X_train": X_train_pca,
            "X_test": X_test_pca,
            "y_train": y_train,
            "y_test": y_test,
        },
    }
    if option == "Worst_Off":
        data_modes = {
            "top_vars": {
                "X_train": X_train_top_vars,
                "X_test": X_test_top_vars,
                "y_train": y_train,
                "y_test": y_test,
            },
            "pca": {
                "X_train": X_train_pca,
                "X_test": X_test_pca,
                "y_train": y_train,
                "y_test": y_test,
            },
        }

    # To make things cleaner, I grouped the models based on their type into different families
    # and for each of them we will use a relatively similar loop and a grid search with cross validation
    # Families: linear_models, tree_based_models, kernel_based_models, neural_networks
    families_trainers = [
        train_baseline_models,
        train_tree_models,
        train_kernel_models,
        train_nn_models,
    ]

    results = []  # we will use this to keep the results of the models
    detailed_results = []  # keeping grid search objects as well

    print("Model Trainer: ", "Training started")

    for trainer in families_trainers:
        family_result, family_detailed_result = trainer(data_modes)
        results.extend(family_result)
        detailed_results.extend(family_detailed_result)

    print("Model Trainer: ", "Training done")

    ## Saving the results
    os.makedirs(output_dir, exist_ok=True)

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_file_address_result, index=False)

    # Saving the detailed results to a pickle file (since it has the grid search objects)
    with open(output_file_address_details, "wb") as f:
        pickle.dump(detailed_results, f)

    print("Model Trainer: ", "results saved. Done.")


if __name__ == "__main__":
    """
    Adding the functionality of execution from command line.
    How to use?

    python src/models/trainer.py {Worst_Off or European_Vanilla}

    It automatically saves the data into data/02_processed/ and data/splitted/ folders.
    """

    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["European_Vanilla", "Worst_Off"])
    args = p.parse_args()
    train_all_models(args.version)
