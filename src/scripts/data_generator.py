import numpy as np
import pandas as pd
from itertools import product
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed

from bs import bs_price
from mc import mc_price


def single_option_price(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    mc_paths: int = 10000,
) -> dict[str, float]:
    """
    Helper function to parallelize the option price calculation.
    It gets one specific set of parameters and returns the option price using both methods.

    Parameters:
    - option_type: Type of option (call or put)
    - S: Underlying asset price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - q: Dividend yield
    - mc_paths: Number of Monte Carlo paths to simulate (default: 10000)

    Returns:
    - A dictionary with this structure:
        {
            "option_type": option_type,
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "q": q,
            "bs_price": bs_price,
            "mc_price": mc_price
        }
    """

    bs = bs_price(option_type, S, K, T, r, sigma, q)
    mc = mc_price(option_type, S, K, T, r, sigma, q, n_paths=mc_paths)
    return {
        "option_type": option_type,
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "q": q,
        "bs_price": bs,
        "mc_price": mc,
    }


def generate_dataset(
    Option_types: list[str],
    S_values: list[float],
    K_values: list[float],
    T_values: list[float],
    r_values: list[float],
    sigma_values: list[float],
    q_values: list[float],
    mc_paths: int = 100000,
    output_csv: str = "../data/options_dataset.csv",
) -> None:
    """
    Generate a dataset of option prices using both Black-Scholes and Monte Carlo methods.
    The dataset includes the following columns:
    - Option_types: Type of option (call or put)
    - S: Underlying asset price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - q: Dividend yield
    - bs_price: Black-Scholes price of the option
    - mc_price: Monte Carlo price of the option

    Parameters:
    - Option_types: List of option types (e.g., ["call", "put"])
    - S_values: List of underlying asset prices
    - K_values: List of strike prices
    - T_values: List of time to expiration values (in years)
    - r_values: List of risk-free interest rates
    - sigma_values: List of volatilities of the underlying asset
    - q_values: List of dividend yields
    - mc_paths: Number of Monte Carlo paths to simulate (default: 100000)
    - output_csv: Path to save the generated dataset (default: "../data/options_dataset.csv")

    Returns:
    - None: The function saves the dataset to a CSV file.
    """
    records = []
    possible_combinations = list(
        product(
            Option_types, S_values, K_values, T_values, r_values, sigma_values, q_values
        )
    )  # all possible combinations of parameters
    n_cpu_cores = multiprocessing.cpu_count()  # get the number of CPU cores
    max_threads_cpu_task = n_cpu_cores - 2  # leave 2 cores free for other tasks
    with Executor(max_workers=max_threads_cpu_task) as executor:
        futures = [
            executor.submit(single_option_price, *combo, mc_paths)
            for combo in possible_combinations
        ]
        records = [feature.result() for feature in as_completed(futures)]

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    generate_dataset(
        Option_types=["call", "put"],  # call, put
        S_values=list(np.linspace(50, 150, 50 + 1)),  # 50, 52, 54, ..., 150
        K_values=[100],  # 100
        T_values=[
            0.08,
            0.25,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
        ],  # 1 month, 3 months, 6 months, 1 year, 1.5 years, 2 years, 2.5 years, 3 years, 4 years, 5 years
        r_values=[0.01, 0.025, 0.05, 0.075, 0.1],  # 1%, 2.5%, 5%, 7.5%, 10%
        sigma_values=[0.1, 0.2, 0.4, 0.6, 0.8],
        q_values=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],  # 0%, 1%, 2%, 3%, 4%, 5%
    )
