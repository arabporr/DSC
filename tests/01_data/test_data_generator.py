from src.data.European_Vanilla import mc_price
from src.data.European_Vanilla import bs_price


def test_mc_close_to_bs():
    bs = bs_price("call", 100, 100, 1, 0.05, 0.2)
    mc = mc_price("call", 100, 100, 1, 0.05, 0.2, n_paths=1000000)
    # Check if the difference between the monte carlo and Black-Scholes prices is less than 1%
    assert abs(mc - bs) / bs < 0.01
