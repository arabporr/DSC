from typing import Literal
import numpy as np


def mc_price_worst_off(
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
) -> float:
    """
    Monte Carlo Simulation to estimate a worst-off option price.
    We know for 1 stock we can simply make the average of the payoffs from n_paths simulated paths
    achived by the risk-neutral expectation of the discounted payoff (GBM model).
    C = e^(-rT) * E[max(ST - K, 0)]
    P = e^(-rT) * E[max(K - ST, 0)]

    ST = S * T * e^(r - q - 0.5 * sigma^2) + sigma * sqrt(T) * Z
    where Z is a standard normal random variable.
    The risk-neutral expectation is approximated by the average of the payoffs from n_paths simulated paths.

    The differemce here however is that we have 2 stocks and we want to take the worst of the two. And
    this two stocks are correlated. So we need to take into account the correlation between the two stocks.
    To do so, we will make the fist stock as before and for the second stock we will use update the formula
    to take into account the correlation between the two stocks.

    ST1 = S1 *  e^((r - q1 - 0.5 * sigma1^2) * T + (sigma1 * sqrt(T) * Z1))
    ST2 = S2 *  e^((r - q2 - 0.5 * sigma2^2) * T + (sigma2 * sqrt(T) * (corr * Z1 + sqrt(1 - corr^2) * Z2)))

    C = max(min((ST1 - S1), (ST2 - S2)), 0)
    P = max(min((S1 - ST1), (S2 - ST2)), 0)


    where:
    - C: call option price
    - P: put option price
    - ST1: spot price at maturity for the first asset
    - ST2: spot price at maturity for the second asset
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
    - n_paths: number of paths to simulate

    Returns:
    - present value of option
    """
    dt = T
    drift1 = dt * (r - q1 - 0.5 * sigma1**2)
    diffusion1 = sigma1 * np.sqrt(dt)

    Z1 = np.random.standard_normal(n_paths)
    ST1 = S1 * np.exp(drift1 + diffusion1 * Z1)

    dt = T
    drift2 = dt * (r - q2 - 0.5 * sigma2**2)
    diffusion2 = sigma2 * np.sqrt(dt)

    Z2 = np.random.standard_normal(n_paths)
    ST2 = S2 * np.exp(drift2 + diffusion2 * (corr * Z1 + np.sqrt(1 - corr**2) * Z2))

    if option_type == "call":
        # Confusion with what said in the word document in the email
        # performance_of_first = np.maximum(ST1/S1, 0.0)
        # performance_of_second = np.maximum(ST2/S2, 0.0)

        performance_of_first = np.maximum(ST1 - K1, 0.0)
        performance_of_second = np.maximum(ST2 - K2, 0.0)
        payoffs = np.minimum(performance_of_first, performance_of_second)
    else:
        performance_of_first = np.maximum(K1 - ST1, 0.0)
        performance_of_second = np.maximum(K2 - ST2, 0.0)
        payoffs = np.minimum(performance_of_first, performance_of_second)

    price = np.exp(-r * T) * payoffs.mean()  # discount factor for risk-free rate
    return price
