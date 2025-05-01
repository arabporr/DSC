import argparse
from src.data.generate import generate_data

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
