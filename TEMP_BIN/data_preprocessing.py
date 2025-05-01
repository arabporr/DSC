import os
import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/options_dataset.csv")

# Rename columns for readability
new_names = {
    "Option_type": "option_type",
    "S": "stock_price",
    "K": "strike_price",
    "T": "time_to_maturity",
    "r": "interest_rate",
    "sigma": "volatility",
    "q": "dividend_yield",
    "bs_price": "black_scholes_price",
    "mc_price": "monte_carlo_price",
}

df.rename(columns=new_names, inplace=True)

# Convert categorical type
df["option_type"] = df["option_type"].astype("category")

# Round prices to 4 decimal places
df = df.round(4)

# Save cleaned data to a new CSV file
if not os.path.exists("../data/cleaned_options_dataset.csv"):
    df.to_csv("../data/cleaned_options_dataset.csv", index=False)
    print("Cleaned data saved to CSV file.")
else:
    print("Cleaned data already exists. No new file created.")

# Define the features and target variable
Numerical_features = [
    "stock_price",
    "strike_price",
    "time_to_maturity",
    "interest_rate",
    "volatility",
    "dividend_yield",
]

Categorical_features = ["option_type"]

Target_column = "black_scholes_price"


# Making a pipeline for preprocessing


def add_new_columns(df):
    """
    Add new columns to the DataFrame as new columns.
    T


    Parameters:
        df (pd.DataFrame): The input DataFrame containing the original features.
        Note: since we are using this function with our own generated dataset, we are already sure that the required columns are present.
        However, to follow the standards and best practices, I am going to add the checker in the beginning of the function.

    Returns:
        pd.DataFrame: The DataFrame with new features added.
    """

    # Check if the required columns are present in the DataFrame
    if "stock_price" not in df.columns:
        raise ValueError("The DataFrame must contain the 'stock_price' column.")
    if "strike_price" not in df.columns:
        raise ValueError("The DataFrame must contain the 'strike_price' column.")
    if "time_to_maturity" not in df.columns:
        raise ValueError("The DataFrame must contain the 'time_to_maturity' column.")
    if "volatility" not in df.columns:
        raise ValueError("The DataFrame must contain the 'volatility' column.")
    if "interest_rate" not in df.columns:
        raise ValueError("The DataFrame must contain the 'interest_rate' column.")
    if "dividend_yield" not in df.columns:
        raise ValueError("The DataFrame must contain the 'dividend_yield' column.")

    df = df.copy()

    if not 0 in df["time_to_maturity"].values:
        df["1_over_T"] = 1 / df["time_to_maturity"]
        df["log_T"] = np.log(df["time_to_maturity"])

    df["sqrt_T"] = np.sqrt(df["time_to_maturity"])
    df["log1p_T"] = np.log1p(df["time_to_maturity"])

    df["variance"] = df["volatility"] ** 2

    if not 0 in df["strike_price"].values:
        df["stock_over_strike"] = df["stock_price"] / df["strike_price"]

    if not 0 in df["stock_price"].values:
        df["strike_over_stock"] = df["strike_price"] / df["stock_price"]

    if not 0 in df["interest_rate"].values:
        df["volatility_over_interest"] = df["volatility"] / df["interest_rate"]
        df["dividend_yield_over_interest"] = df["dividend_yield"] / df["interest_rate"]
        df["stock_price_over_interest"] = df["stock_price"] / df["interest_rate"]

    if not 0 in df["dividend_yield"].values:
        df["volatility_over_dividend"] = df["volatility"] / df["dividend_yield"]
        df["interest_rate_over_dividend"] = df["interest_rate"] / df["dividend_yield"]
        df["stock_price_over_dividend"] = df["stock_price"] / df["dividend_yield"]

    if not 0 in df["volatility"].values:
        df["interest_rate_over_volatility"] = df["interest_rate"] / df["volatility"]
        df["dividend_yield_over_volatility"] = df["dividend_yield"] / df["volatility"]
        df["stock_price_over_volatility"] = df["stock_price"] / df["volatility"]

    return df


# Adding new columns to the DataFrame
add_new_cols = FunctionTransformer(func=add_new_columns, validate=False)

# Adding polynomial features
add_poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

numerical_pipeline = make_pipeline(
    add_new_cols,
    add_poly,
    StandardScaler(),
)

categorical_pipeline = Pipeline(
    [
        ("onehot", OneHotEncoder(drop="if_binary")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numerical", numerical_pipeline, Numerical_features),
        ("categorical", categorical_pipeline, Categorical_features),
    ]
)

df_temp_1 = add_new_cols.fit_transform(df[Numerical_features])
df_temp_2 = pd.DataFrame(
    add_poly.fit_transform(df_temp_1),
    columns=add_poly.get_feature_names_out(df_temp_1.columns),
)
df_temp_2.columns = df_temp_2.columns.str.replace(" ", "_times_", regex=False)

num_feature_names = df_temp_2.columns.tolist()


df_temp_1 = pd.DataFrame(
    categorical_pipeline.fit_transform(df[Categorical_features]).toarray(),
    columns=categorical_pipeline.get_feature_names_out(Categorical_features),
)
cat_feature_names = df_temp_1.columns.tolist()


all_feature_names = list(num_feature_names) + list(cat_feature_names)
print("Pipeline tested. Output features are:\n", all_feature_names)


processed_data = preprocessor.fit_transform(
    df[Numerical_features + Categorical_features]
)
processed_data = pd.DataFrame(processed_data, columns=all_feature_names, index=df.index)
processed_data[Target_column] = df[Target_column]

if not os.path.exists("../data/processed_data.csv"):
    processed_data.to_csv("../data/processed_options_dataset.csv")
    print("Processed data saved to CSV file.")
else:
    print("Processed data already exists. No new file created.")


# Split the data into training and testing sets
try:
    data = processed_data.copy()
except:
    data = pd.read_csv("../data/processed_options_dataset.csv", index_col=0)

# making sure that there is no information leakage in the dataset
if "monte_carlo_price" in data.columns:
    data = data.drop(columns=["monte_carlo_price"])
if "mc_price" in data.columns:
    data = data.drop(columns=["mc_price"])
if "bs_price" in data.columns:
    data = data.drop(columns=["bs_price"])


X = data.drop(columns=["black_scholes_price"])
y = data["black_scholes_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2025, shuffle=True
)


if not os.path.exists("../data/training_data.csv"):
    training_data = pd.concat([X_train, y_train], axis=1)
    training_data.to_csv("../data/training_data.csv", index=False)
    print("Training data saved to CSV file.")
else:
    print("Training data already exists. No new file created.")

if not os.path.exists("../data/testing_data.csv"):
    testing_data = pd.concat([X_test, y_test], axis=1)
    testing_data.to_csv("../data/testing_data.csv", index=False)
    print("Testing data saved to CSV file.")
else:
    print("Testing data already exists. No new file created.")

print("Data preprocessing completed successfully.")
