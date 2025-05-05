import os
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics between true and predicted values.

    Parameters:
      - y_true (np.ndarry): the actual values
      - y_pred (np.ndarray): the predicted values

    Returns a dict with this structure:
      - rmse: Root Mean Squared Error
      - mae: Mean Absolute Error
      - r2: R-squared
      - mape: Mean Absolute Percentage Error
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def evaluate_from_detailed(
    detailed_results_path: str,
    test_csv_path: str,
    target_col: str,
    output_csv: str = None,
) -> pd.DataFrame:
    """
    Load a detailed_results.pkl containing GridSearchCV objects,
    apply each best_estimator_ to the test set,
    compute regression metrics, and return a DataFrame of results.

    Optionally save the results to CSV if output_csv is provided.
    """
    # Load detailed results
    with open(detailed_results_path, "rb") as f:
        detailed_results = pickle.load(f)

    # Load test data
    df_test = pd.read_csv(test_csv_path)
    y_true = df_test[target_col].values
    X_test = df_test.drop(columns=[target_col])

    records = []
    for rec in detailed_results:
        gs = rec.get("gs_object") or rec.get("grid_search")
        model = gs.best_estimator_
        y_pred = model.predict(X_test)

        metrics = compute_regression_metrics(y_true, y_pred)
        record = {
            "family": rec.get("family"),
            "model": rec.get("model"),
            "data_mode": rec.get("data_mode"),
            **metrics,
        }
        records.append(record)

    df_metrics = pd.DataFrame(records)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_metrics.to_csv(output_csv, index=False)

    return df_metrics


def plot_pred_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
) -> None:
    """
    Scatter plot of predicted vs. actual values with 45-degree reference line.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, "--k")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_residual_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
) -> None:
    """
    Plot the distribution of residuals (errors = y_true - y_pred).
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, stat="density")
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
