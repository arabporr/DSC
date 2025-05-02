import argparse

from src.data.generate import generate_data
from src.features.preprocess import preprocessor
from src.models.trainer import train_all_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for options pricing.")
    parser.add_argument(
        "option",
        choices=["European_Vanilla", "Worst_Off"],
        help="Type of option data to generate.",
    )
    args = parser.parse_args()

    print(f"Generating data for {args.option}...")
    generate_data(args.option)
    print(f"Data generation for {args.option} completed.")
    print(f"Preprocessing data for {args.option}...")
    preprocessor(args.option)
    print(f"Data preprocessing for {args.option} completed.")
    print(f"Training models for {args.option}...")
    train_all_models(args.option)
    print(f"Model training for {args.option} completed.")
