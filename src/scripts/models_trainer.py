import os
import time
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# Data loading
training_data = pd.read_csv("../data/training_data.csv")
test_data = pd.read_csv("../data/testing_data.csv")

X_train = training_data.iloc[:, :-1]
y_train = training_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]


# Data preparation
# Identifying columns with almost constant values
Columns_variance_filter = X_train.loc[:, X_train.var() > 1e-2].columns.tolist()

# Identifying columns with low correlation to the target variable
df = X_train.copy()
df["y"] = y_train
corr = df.corr()
Columns_correlation_filter = corr["y"].abs()[corr["y"].abs() > 0.05].index.tolist()

# Identifying top 40 features using mutual information regression
Feature_selection = SelectKBest(mutual_info_regression, k=40).fit(X_train, y_train)
Top_40_columns = X_train.columns[Feature_selection.get_support(indices=True)]

# Getting the intersection of the methods above
top_variables = set(Top_40_columns).intersection(
    set(Columns_variance_filter).intersection(set(Columns_correlation_filter))
)

X_train_top_vars = X_train[list(top_variables)]
X_test_top_vars = X_test[list(top_variables)]


# Also making a smaller set of features using PCA
pca_full = PCA(random_state=2025)
pca_full.fit(X_train)

# Adding the amount of variance explained by each component
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# Setting a threshold for the amount of variance to be explained
# Here I just tried different values since 95 till 99 were in 20-ish range
# I decided to go with 98%, not too crazy and not too low
Var_threshold = 0.98
Number_of_components = np.where(cum_var > Var_threshold)[0][
    0
]  # this the index of the first component that explains more than 98% of the variance

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
    },
    "top_vars": {
        "X_train": X_train_top_vars,
        "X_test": X_test_top_vars,
    },
    "pca": {
        "X_train": X_train_pca,
        "X_test": X_test_pca,
    },
}


results = []  # we will use this to keep the results of the models
detailed_results = []  # we will use this to keep grid search results objects as well

## Models to train
# To make things cleaner, I will group the models based on their type into different families
# and for each of them we will use a relatively similar loop and a grid search with cross validation
# Families: linear_models, tree_based_models, kernel_based_models, neural_networks

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


# Second family: tree_based_models
family_name = "tree_models"
print(f"Running {family_name} family:\n")

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
        X_train_temp = data_modes[data_mode]["X_train"]
        X_test_temp = data_modes[data_mode]["X_test"]

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
            full_len = len(X_train_temp)
            sample_size = int(ratio * full_len)
            sample_indices = np.random.choice(full_len, size=sample_size, replace=False)
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


# Fourth family: neural_networks
family_name = "neural_networks"
print(f"Running {family_name} family:\n")

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

        # no complexity, we go all in and test all the data
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


## Saving the results
os.makedirs("../results", exist_ok=True)

df_res = pd.DataFrame(results)
df_res.to_csv("../results/model_comparison.csv", index=False)

# Saving the detailed results to a pickle file (since it has the grid search objects)
with open("../results/detailed_results.pkl", "wb") as f:
    pickle.dump(detailed_results, f)
