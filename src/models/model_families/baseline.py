import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge


results = []  # we will use this to keep the results of the models
detailed_results = []  # we will use this to keep grid search results objects as well

# First family: linear_models
family_name = "linear_models"
print(f"Running {family_name} family:\n")

# Grid of hyperparameters for each model
linear_models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "grid": {"fit_intercept": [True, False]},
    },
    "Ridge": {"model": Ridge(), "grid": {"alpha": [0.1, 1.0, 10.0]}},
}

for model_name, settings in linear_models.items():
    for data_mode, data in track(
        data_modes.items(),
        description=f"Running different data modes for {model_name} model:",
    ):
        X_train_temp = data_modes[data_mode]["X_train"]
        X_test_temp = data_modes[data_mode]["X_test"]

        model = settings["model"]
        param_grid = settings["grid"]

        # Main part is here, we use grid search with cross validation
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

        start = time.time()
        gs.fit(X_train_temp, y_train)
        elapsed = time.time() - start

        best = gs.best_estimator_
        train_rmse = root_mean_squared_error(y_train, best.predict(X_train_temp))
        test_rmse = root_mean_squared_error(y_test, best.predict(X_test_temp))

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
