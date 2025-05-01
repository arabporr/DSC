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
