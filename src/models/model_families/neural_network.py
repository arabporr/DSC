import time
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error

from sklearn.neural_network import MLPRegressor


def train_nn_models(data_modes: dict) -> List[dict]:
    results = []  # we will use this to keep the results of the models
    detailed_results = []  # keeping grid search objects as well

    # Fourth family: neural_networks
    family_name = "neural_networks"
    print(f"\n\n\n\nRunning {family_name} family:\n")

    # Hyperparameters for the neural networks
    neural_networks = {
        "MLPRegressor": {
            "model": MLPRegressor(max_iter=100, random_state=2025),
            "grid": {
                "hidden_layer_sizes": [
                    (10,),
                    (50,),
                    (
                        20,
                        20,
                        20,
                        20,
                    ),
                    (
                        50,
                        50,
                    ),
                ],
                "learning_rate": ["constant", "adaptive"],
            },
        },
    }

    for model_name, settings in neural_networks.items():
        print(f"Running {model_name} model:\n")
        for data_mode, data in track(
            data_modes.items(), description=f"data mode {data_mode}"
        ):

            X_train_mode = data["X_train"]
            X_test_mode = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            model = settings["model"]
            param_grid = settings["grid"]

            # Main part, like previous groups
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

            # no complexity, we go all in and test all the data
            start = time.time()
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
