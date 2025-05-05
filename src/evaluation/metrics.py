import os
import pickle

from typing import Literal

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def models_eval_selection(option: Literal["European_Vanilla", "Worst_Off"]) -> None:
    """
    Finds the best model based on test root mean square error for the given option data type.
    It then plots the stats for all models to support its decision. Then to verify that the model
    is not overfitted over or have big biases, it plots the residuals of train and test set predictions.
    Finally, to make it easier to use the best model later, it saves the best model object in the appropriate folder.

    Parameters:
      - option (Literal[European_Vanilla, Worst_Off]): option data type

    Return:
        None: It makes some plots and saves data into files.
    """
    # file paths
    if option == "European_Vanilla":
        model_comparison_file_path = "results/European_Vanilla/model_comparison.csv"
        models_logs_file_path = "results/European_Vanilla/models_log.pkl"
        results_directory = "results/European_Vanilla/"

        data_modes_file_path = "data/03_splitted/European_Vanilla/data_modes.pkl"
        best_model_save_path = "results/European_Vanilla/best_model.plk"
    elif option == "Worst_Off":
        model_comparison_file_path = "results/Worst_Off/model_comparison.csv"
        models_logs_file_path = "results/Worst_Off/models_log.pkl"
        results_directory = "results/Worst_Off/"

        data_modes_file_path = "data/03_splitted/Worst_Off/data_modes.pkl"
    else:
        raise ValueError(
            "Invalid option. Choose either 'European_Vanilla' or 'Worst_Off'."
        )

    # Loading results from training
    model_comparison = pd.read_csv(model_comparison_file_path)
    with open(models_logs_file_path, "rb") as f:
        models_logs = pickle.load(f)
    with open(data_modes_file_path, "rb") as f:
        data_modes = pickle.load(f)

    # Finding best model
    best_model_idx = model_comparison["test_rmse"].idxmin()
    best_model_info = model_comparison.loc[best_model_idx]

    print(f"Best model for {option} based on lowest test RMSE:")
    print(best_model_info.to_frame().T)

    # Plots
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=model_comparison, x="model", y="test_rmse", hue="data_mode")
    ax.set_title("Test RMSE by Model and Data Mode")
    ax.set_ylabel("Test RMSE")
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "Evaluation_Bar_Plot.png"))
    plt.show()

    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(
        data=model_comparison,
        x="train_rmse",
        y="test_rmse",
        hue="model",
        style="data_mode",
        s=100,
    )
    ax.plot(
        [model_comparison["train_rmse"].min(), model_comparison["train_rmse"].max()],
        [model_comparison["train_rmse"].min(), model_comparison["train_rmse"].max()],
        ls="--",
        color="gray",
    )
    ax.set_title("Train vs Test RMSE")
    ax.set_xlabel("Train RMSE")
    ax.set_ylabel("Test RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "Evaluation_Scatter_Plot"))
    plt.show()

    # loading the best model
    best_model_object = models_logs[best_model_idx].get("gs_object")
    data = data_modes[models_logs[best_model_idx].get("data_mode")]

    # Making predictions and the residuals
    y_train = data["y_train"]
    y_pred_train = best_model_object.predict(data["X_train"])
    y_test = data["y_test"]
    y_pred_test = best_model_object.predict(data["X_test"])

    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    # Plotting them
    plt.figure(figsize=(12, 8))
    plt.hist(residuals_train, bins=100, alpha=0.5, label="Train residuals")
    plt.hist(residuals_test, bins=100, alpha=0.5, label="Test residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Overlapping Distribution of Residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "Best_Model_Residuals.png"))
    plt.show()

    # Saving the best model object for later use
    model_name = best_model_info["model"]
    data_modes_type = best_model_info["data_mode"]
    best_model_file_name = f"best_model_{model_name}_{data_modes_type}.pkl"
    best_model_save_path = os.path.join(results_directory, best_model_file_name)
    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
    with open(best_model_save_path, "wb") as f:
        pickle.dump(best_model_object, f)
