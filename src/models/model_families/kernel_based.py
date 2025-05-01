import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


results = []  # we will use this to keep the results of the models
detailed_results = []  # we will use this to keep grid search results objects as well

# Third family: kernel_based_models
family_name = "kernel_models"
print(f"Running {family_name} family:\n")

# The grid of hyperparameters for each model
kernel_models = {
    "SVR": {
        "model": SVR(),
        "grid": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0, 10.0],
        },
    },
    "KNeighborsRegressor": {
        "model": KNeighborsRegressor(),
        "grid": {
            "n_neighbors": [3, 5, 10],
            "leaf_size": [10, 20],
            "weights": ["uniform", "distance"],
        },
    },
}

for model_name, settings in kernel_models.items():
    print(f"Running {model_name} model:\n")
    for data_mode, data in track(
        data_modes.items(),
        description=f"Running different data modes for {model_name} model:",
    ):
        if data_mode == "all_variables":
            # skipping this combination as it takes too long
            continue

        size_reduction_factor = 0.1
        size_limit = int(len(X_train) * size_reduction_factor)

        X_train_temp = data_modes[data_mode]["X_train"]
        X_test_temp = data_modes[data_mode]["X_test"]

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

        # Here again, like previous group, we have a new version
        # This time since one of the algorithms is slow (SVR)
        # we will save some time by sampling 40% of the data for faster computation
        # regardless of the data mode, since the issue is with the algorithm itself
        start = time.time()
        if model_name == "SVR":
            ratio = 0.4
            full_len = len(X_train_temp)
            sample_size = int(ratio * full_len)
            sample_indices = np.random.choice(full_len, size=sample_size, replace=False)
            # The data type for the PCA data is different from the other two, its a numpy array
            # so we need to use indexing instead of iloc
            if data_mode == "pca":
                gs.fit(X_train_temp[sample_indices], y_train[sample_indices])
            else:
                gs.fit(X_train_temp.iloc[sample_indices], y_train.iloc[sample_indices])
        else:
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
