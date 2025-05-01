import argparse
from src.models.trainer import train_all_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for options pricing.")
    parser.add_argument(
        "option",
        choices=["European_Vanilla", "Worst_Off"],
        help="Type of option data to train models on.",
    )
    args = parser.parse_args()

    print(f"Training models for {args.option}...")
    train_all_models(args.option)
    print(f"Model training for {args.option} completed.")
