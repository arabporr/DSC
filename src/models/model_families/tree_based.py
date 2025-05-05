import time
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_tree_models(data_modes: dict, models_grids: dict = {}) -> List[dict]:
    """
    The helper function to train and find the best hyperparameters for a specific family of models. Here: tree based models
    It receives data and model congifs in the input and tries a grid search on models and their corresponding space of parameters
    To make it reuseable after initial testing of all models, it has the model_gird input option which one can use to search with different
    settings within a family.

    Args:
        data_modes (dict): the dictionary of data, containing data modes (all_var, top_var, or pca)
        models_grids (dict, optional): the dictionary of pair of models and their corresponding parameter space

    Returns:
        List[dict]: the list of results for the models trained, containing one numerical and one with the gridsearch objects (results and models_log)
    """
    results = []  # we will use this to keep the results of the models
    models_log = []  # keeping grid search objects as well

    # Second family: tree_based_models
    family_name = "tree_models"
    print(f"\n\n\n\nRunning {family_name} family:\n")

    if models_grids == {}:
        # The grid of hyperparameters for each model
        tree_models = {
            "RandomForestRegressor": {
                "model": RandomForestRegressor(random_state=2025),
                "grid": {
                    "n_estimators": [10, 20, 40],
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [0.1, 100],
                    "max_features": ["sqrt"],
                },
            },
            "XGBRegressor": {
                "model": XGBRegressor(
                    tree_method="hist", random_state=2025, verbosity=1
                ),
                "grid": {
                    "n_estimators": [10, 20, 40],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4],
                    "max_depth": [1, 2, 5, 10],
                },
            },
        }
    else:
        tree_models = models_grids

    for model_name, settings in tree_models.items():
        print(f"Running {model_name} model:\n")
        for data_mode, data in track(
            data_modes.items(),
            description=f"Running different data modes for {model_name} model:",
        ):
            X_train_mode = data["X_train"]
            X_test_mode = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            model = settings["model"]
            param_grid = settings["grid"]

            # Main part, like previous group.
            # score models with negative root mean squared error
            # with 3 folds of cross validation
            # and in parallel using all available cores (n_jobs=-1)
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=3,
                n_jobs=-2,
                verbose=10,
            )

            # Here we have a new version and there is a reason for that
            # since tree based models are more computationally expensive
            # it took too long to run the grid search on the full dataset
            # so I decided to sample 40% of the data for faster computation when training with all variables
            # the reason behind 40%: the amount of data looks good enough (basically personal choice)
            start = time.time()
            if data_mode == "all_variables":
                # sampling 40% of the data for faster computation
                ratio = 0.4
                full_len = len(X_train_mode)
                sample_size = int(ratio * full_len)
                sample_indices = np.random.choice(
                    full_len, size=sample_size, replace=False
                )

                gs.fit(X_train_mode.iloc[sample_indices], y_train.iloc[sample_indices])
            else:
                gs.fit(X_train_mode, y_train)

            elapsed = time.time() - start

            best = gs.best_estimator_
            y_pred_train = best.predict(X_train_mode)
            y_pred_test = best.predict(X_test_mode)
            train_rmse = root_mean_squared_error(y_train, y_pred_train)
            test_rmse = root_mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

            results.append(
                {
                    "family": family_name,
                    "model": model_name,
                    "data_mode": data_mode,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "train_mae": train_mae,
                    "test_mae": test_mae,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_mape": train_mape,
                    "test_mape": test_mape,
                    "time_s": elapsed,
                    "best_estimator": best,
                }
            )

            models_log.append(
                {
                    "family": family_name,
                    "model": model_name,
                    "data_mode": data_mode,
                    "time_s": elapsed,
                    "gs_object": gs,
                }
            )

    return [results, models_log]
