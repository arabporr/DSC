from typing import Literal
import numpy as np
from scipy.stats import norm


def bs_price(
    option_type: Literal["call", "put"],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """
    European option price with Black-Scholes formula:
    C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
    P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
    d1 = (ln(S/K) + (r - q + 0.5 * sigma^2)T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    where:
    - C: call option price
    - P: put option price
    - S: spot price
    - K: strike price
    - T: time to maturity (years)
    - r: risk-free rate (annual)
    - sigma: volatility (annual)
    - q: dividend yield (annual)


    Parameters:
    - option_type: "call" or "put"
    - S: spot price
    - K: strike price
    - T: time to maturity (years)
    - r: risk-free rate (annual)
    - sigma: volatility (annual)
    - q: dividend yield (annual)

    Returns:
      present value of option
    """
    if T <= 0:
        # immediate payoff
        if option_type == "call":
            payoff = max(S - K, 0.0)
        else:
            max(K - S, 0.0)
        return payoff

    d1 = (np.log(S / K) + T * (r - q + 0.5 * (sigma**2))) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc_q = np.exp(-q * T)  # discount factor for dividend yield
    disc_r = np.exp(-r * T)  # discount factor for risk-free rate

    if option_type == "call":
        return disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
    else:
        return disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1)


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
