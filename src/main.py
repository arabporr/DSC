import argparse

from src.data.generate import generate_data
from src.features.preprocess import preprocessor
from src.models.trainer import train_all_models
from src.evaluation.metrics import models_eval_selection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for options pricing.")
    parser.add_argument(
        "option",
        choices=["European_Vanilla", "Worst_Off"],
        help="Type of option data to generate.",
    )
    args = parser.parse_args()
    option_data_type = args.option

    print(f"Generating data for {option_data_type}...")
    generate_data(option_data_type)
    print(f"Data generation for {option_data_type} completed.")
    print(f"Preprocessing data for {option_data_type}...")
    preprocessor(option_data_type)
    print(f"Data preprocessing for {option_data_type} completed.")
    print(f"Training models for {option_data_type}...")
    train_all_models(option_data_type)
    print(f"Model training for {option_data_type} completed.")
    print(f"Evaluating models for {option_data_type}...")
    models_eval_selection(option_data_type)
    print(f"Models evaluation for {option_data_type} completed.")
    print("The project execution finished successfully!")
