import argparse
from src.evaluation.metrics import models_eval_selection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for options pricing.")
    parser.add_argument(
        "option",
        choices=["European_Vanilla", "Worst_Off"],
        help="Type of option data to preprocess.",
    )
    args = parser.parse_args()

    print(f"Evaluating models for {args.option}...")
    models_eval_selection(args.option)
    print(f"Models evaluation for {args.option} completed.")
