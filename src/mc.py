from typing import Literal
import numpy as np


def mc_price(
    option_type: Literal["call", "put"],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 10000,  # number of paths to simulate, to make error 1/sqrt(n_paths) = 0.01 or 1%
) -> float:
    """
    Monte Carlo Simulation to estimate a European option price.
    Assumes a log-normal distribution of the underlying asset price at maturity. (GBM model)
    The option price is given by the risk-neutral expectation of the discounted payoff:
    C = e^(-rT) * E[max(ST - K, 0)]
    P = e^(-rT) * E[max(K - ST, 0)]
    ST = S * T * e^(r - q - 0.5 * sigma^2) + sigma * sqrt(T) * Z
    where Z is a standard normal random variable.
    The risk-neutral expectation is approximated by the average of the payoffs from n_paths simulated paths.

    where:
    - C: call option price
    - P: put option price
    - ST: spot price at maturity
    - S: spot price
    - K: strike price
    - T: time to maturity (years)
    - r: risk-free rate (annual)
    - sigma: volatility (annual)
    - q: dividend yield (annual)


    option_type: "call" or "put"
    - S: spot price
    - K: strike price
    - T: time to maturity (years)
    - r: risk-free rate (annual)
    - sigma: volatility (annual)
    - q: dividend yield (annual)
    - n_paths: number of paths to simulate

    Returns:
    - present value of option
    """
    dt = T
    drift = dt * (r - q - 0.5 * sigma**2)
    diffusion = sigma * np.sqrt(dt)

    Z = np.random.standard_normal(n_paths)
    ST = S * np.exp(drift + diffusion * Z)
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    price = np.exp(-r * T) * payoffs.mean()  # discount factor for risk-free rate
    return price
