import argparse
from src.features.preprocess import preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for options pricing.")
    parser.add_argument(
        "option",
        choices=["European_Vanilla", "Worst_Off"],
        help="Type of option data to preprocess.",
    )
    args = parser.parse_args()

    print(f"Preprocessing data for {args.option}...")
    preprocessor(args.option)
    print(f"Data preprocessing for {args.option} completed.")
