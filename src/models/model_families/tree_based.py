import time
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_tree_models(data_modes: dict) -> List[dict]:
    results = []  # we will use this to keep the results of the models
    detailed_results = []  # keeping grid search objects as well

    # Second family: tree_based_models
    family_name = "tree_models"
    print(f"\n\n\n\nRunning {family_name} family:\n")

    # The grid of hyperparameters for each model
    tree_models = {
        "RandomForestRegressor": {
            "model": RandomForestRegressor(random_state=2025),
            "grid": {"n_estimators": [10, 40], "max_depth": [5, 10]},
        },
        "XGBRegressor": {
            "model": XGBRegressor(tree_method="hist", random_state=2025, verbosity=1),
            "grid": {
                "n_estimators": [10, 40],
                "learning_rate": [0.01, 0.1],
                "max_depth": [2, 6],
            },
        },
    }

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
                n_jobs=-1,
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
            train_rmse = root_mean_squared_error(y_train, best.predict(X_train_mode))
            test_rmse = root_mean_squared_error(y_test, best.predict(X_test_mode))

            results.append(
                {
                    "family": family_name,
                    "model": model_name,
                    "data_mode": data_mode,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "time_s": elapsed,
                    "best_estimator": best,
                }
            )

            detailed_results.append(
                {
                    "family": family_name,
                    "model": model_name,
                    "data_mode": data_mode,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "time_s": elapsed,
                    "gs_object": gs,
                }
            )

    return [results, detailed_results]
