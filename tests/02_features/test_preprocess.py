import numpy as np
import pandas as pd


df = pd.read_csv("data/02_processed/European_Vanilla_processed_dataset.csv")

assert df.isna().sum().sum() == 0
assert np.isfinite(df.select_dtypes(include=[float])).all().all()

subset = [
    "stock_price",
    "strike_price",
    "time_to_maturity",
    "interest_rate",
    "volatility",
    "dividend_yield",
]
assert df.duplicated(subset=subset).sum() == (len(df) / 2)
