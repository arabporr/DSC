from typing import Literal
import argparse

import numpy as np
import pandas as pd
from itertools import product
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed

from src.data.European_Vanilla import bs_price, mc_price
from src.data.Worst_Off import mc_price_worst_off


def single_option_price_europian_vanilla(
    option_type: Literal["call", "put"],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 10000,
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
    - n_paths: Number of Monte Carlo paths to simulate (default: 10000)

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
    mc = mc_price(option_type, S, K, T, r, sigma, q, n_paths)
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


def generate_european_vanilla(
    Option_types: list[str],
    S_values: list[float],
    K_values: list[float],
    T_values: list[float],
    r_values: list[float],
    sigma_values: list[float],
    q_values: list[float],
    n_paths: int = 10000,
) -> pd.DataFrame:
    """
    Generate a dataset of option prices using both Black-Scholes and Monte Carlo methods.
    To make the process faster, we use multiprocessing to parallelize the calculations.
    (I used it here since I know the calculations are independent)

    The dataset includes the following columns:
    - Option_type: Type of option (call or put)
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
    - n_paths: Number of Monte Carlo paths to simulate (default: 10000)

    Returns:
    - padf.DataFrame: A DataFrame containing the generated dataset.
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
            executor.submit(single_option_price_europian_vanilla, *combo, n_paths)
            for combo in possible_combinations
        ]
        records = [feature.result() for feature in as_completed(futures)]

    generated_dataset = pd.DataFrame(records)
    return generated_dataset


def single_option_price_worst_off(
    option_type: Literal["call", "put"],
    S1: float,
    S2: float,
    K1: float,
    K2: float,
    sigma1: float,
    sigma2: float,
    q1: float = 0.0,
    q2: float = 0.0,
    corr: float = 0.0,
    T: float = 1.0,
    r: float = 0.05,
    n_paths: int = 10000,  # number of paths to simulate, to make error 1/sqrt(n_paths) = 0.01 or 1%
) -> dict[str, float]:
    """
    Helper function to parallelize the option price calculation.
    It gets one specific set of parameters and returns the option price using monte carlo method.

    Parameters:
    - option_type: "call" or "put"
    - S1: spot price of the first asset
    - S2: spot price of the second asset
    - K1: strike price of the first asset
    - K2: strike price of the second asset
    - sigma1: volatility of the first asset
    - sigma2: volatility of the second asset
    - q1: dividend yield of the first asset (annual)
    - q2: dividend yield of the second asset (annual)
    - corr: correlation between the two assets
    - T: time to maturity (years)
    - r: risk-free rate (annual)
    - n_paths: Number of Monte Carlo paths to simulate (default: 10000)

    Returns:
    - A dictionary with this structure:
        {
            "option_type": option_type,
            "S1": S1,
            "S2": S2,
            "K1": K1,
            "K2": K2,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "q1": q1,
            "q2": q2,
            "corr": corr,
            "T": T,
            "r": r,
            "price": price,
        }
    """

    price = mc_price_worst_off(
        option_type,
        S1,
        S2,
        K1,
        K2,
        sigma1,
        sigma2,
        q1,
        q2,
        corr,
        T,
        r,
        n_paths,
    )

    return {
        "option_type": option_type,
        "S1": S1,
        "S2": S2,
        "K1": K1,
        "K2": K2,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "q1": q1,
        "q2": q2,
        "corr": corr,
        "T": T,
        "r": r,
        "price": price,
    }


def generate_worst_off(
    Option_types: list[str],
    S1_values: list[float],
    S2_values: list[float],
    K1_values: list[float],
    K2_values: list[float],
    sigma1_values: list[float],
    sigma2_values: list[float],
    q1_values: list[float],
    q2_values: list[float],
    corr_values: list[float],
    T_values: list[float],
    r_values: list[float],
    n_paths: int = 10000,
) -> pd.DataFrame:
    """
    Generate a dataset of option prices using Monte Carlo method for the worst-off option.
    To make the process faster, we use multiprocessing to parallelize the calculations.
    (I used it here since I know the calculations are independent)

    The dataset includes the following columns:
    - Option_type: Type of option (call or put)
    - S: Underlying asset price of the first asset
    - S: Underlying asset price of the second asset
    - K: Strike price of the first asset
    - K: Strike price of the second asset
    - sigma: Volatility of the underlying asset of the first asset
    - sigma: Volatility of the underlying asset of the second asset
    - q: Dividend yield of the first asset (annual)
    - q: Dividend yield of the second asset (annual)
    - corr: Correlation between the two assets
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - mc_price: Monte Carlo price of the option

    Parameters:
    - Option_types: List of option types (e.g., ["call", "put"])
    - S1_values: List of underlying asset prices of the first asset
    - S2_values: List of underlying asset prices of the second asset
    - K1_values: List of strike prices of the first asset
    - K2_values: List of strike prices of the second asset
    - sigma1_values: List of volatilities of the underlying asset of the first asset
    - sigma2_values: List of volatilities of the underlying asset of the second asset
    - q1_values: List of dividend yields of the first asset (annual)
    - q2_values: List of dividend yields of the second asset (annual)
    - corr_values: List of correlation values
    - T_values: List of time to expiration values (in years)
    - r_values: List of risk-free interest rates
    - n_paths: Number of Monte Carlo paths to simulate (default: 100000)

    Returns:
    - padf.DataFrame: A DataFrame containing the generated dataset.
    """

    records = []
    possible_combinations = list(
        product(
            Option_types,
            S1_values,
            S2_values,
            K1_values,
            K2_values,
            sigma1_values,
            sigma2_values,
            q1_values,
            q2_values,
            corr_values,
            T_values,
            r_values,
        )
    )  # all possible combinations of parameters
    n_cpu_cores = multiprocessing.cpu_count()  # get the number of CPU cores
    max_threads_cpu_task = n_cpu_cores - 2  # leave 2 cores free for other tasks
    with Executor(max_workers=max_threads_cpu_task) as executor:
        futures = [
            executor.submit(single_option_price_worst_off, *combo, n_paths)
            for combo in possible_combinations
        ]
        records = [feature.result() for feature in as_completed(futures)]

    generated_dataset = pd.DataFrame(records)
    return generated_dataset


def generate_data(
    option: Literal["European_Vanilla", "Worst_Off"],
    European_Vanilla_output_csv: str = "data/01_raw/European_Vanilla_dataset.csv",
    Worst_Off_output_csv: str = "data/01_raw/Worst_Off_dataset.csv",
) -> None:
    """
    Wrapper function to generate data for different types of options.

    Parameters:
        option (str): The type of option to generate data for. Can be "European_Vanilla" or "Worst_Off".
    Returns:
        None: This function does not return anything. It generates a dataset and saves it to a CSV file.
    """
    if option == "European_Vanilla":

        dataset = generate_european_vanilla(
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

        dataset.to_csv(European_Vanilla_output_csv, index=False)
        print(f"Saved {len(dataset)} rows to {European_Vanilla_output_csv}")

    elif option == "Worst_Off":
        dataset = generate_worst_off(
            Option_types=["call", "put"],  # call, put
            S1_values=list(np.linspace(80, 120, 8 + 1)),  # 80, 85, ..., 120
            S2_values=list(np.linspace(80, 120, 8 + 1)),  # 80, 85, ..., 120
            K1_values=[100],  # 100
            K2_values=[100],  # 100
            sigma1_values=[0.1, 0.4, 0.8],
            sigma2_values=[0.1, 0.4, 0.8],
            q1_values=[0.01, 0.02, 0.05],  # 0%, 2%, 5%
            q2_values=[0.01, 0.02, 0.05],  # 0%, 2%, 5%
            corr_values=[0.01, 0.2, 0.5, 0.8],  # correlation coefficients
            T_values=[
                0.08,
                0.5,
                1.0,
                2.0,
                5.0,
            ],  # 1 month, 6 months, 1 year, 2 years, 5 years
            r_values=[0.01, 0.025, 0.05, 0.1],  # 1%, 2.5%, 5%, 10%
        )

        dataset.to_csv(Worst_Off_output_csv, index=False)
        print(f"Saved {len(dataset)} rows to {Worst_Off_output_csv}")

    else:
        raise ValueError("option_type must be either 'European' or 'Worst_Off'")


if __name__ == "__main__":
    """
    Adding the functionality of execution from command line.
    How to use?

    python src/data/generate.py {Worst_Off or European_Vanilla}

    It automatically saves the data into data/01_raw/ folder.
    """

    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["European_Vanilla", "Worst_Off"])
    args = p.parse_args()
    generate_data(args.version)
