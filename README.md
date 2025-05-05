# Option Pricing Data Science Challenge

This project implements a comprehensive solution for pricing European options using both analytical and numerical approaches, with a focus on machine learning applications. The solution includes vanilla European options and extends to more complex exotic options like the European Worst-Off option.

## Project Structure

```
.
├── data/                  # Data storage directory
├── notebooks/            # Jupyter notebooks for analysis and development
│   ├── 01_EDA_European_Vanilla.ipynb
│   ├── 02_Data_Transformation.ipynb
│   ├── 03_Models.ipynb
│   └── 04_EDA_Worst_Off.ipynb
├── results/              # Output results and visualizations
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── features/        # Feature engineering
│   ├── models/          # ML model implementations
│   ├── scripts/         # Utility scripts
│   └── evaluation/      # Model evaluation code
├── tests/               # Test suite
├── requirements.txt     # Python dependencies
└── setup.py            # Project setup configuration
```

## Features

- **Analytical Pricing**: Implementation of the Black-Scholes formula for European vanilla options
- **Numerical Methods**: Monte Carlo simulation for option pricing
- **Machine Learning Models**: Various regression models for price prediction
- **Exotic Options**: Extension to European Worst-Off options
- **Comprehensive Analysis**: Data exploration, visualization, and model evaluation

## Technical Implementation Flow

### 1. Data Generation Pipeline

#### 1.1 Black-Scholes Implementation
- Implemented in `src/data/black_scholes.py`
- Core components:
  - d1 and d2 calculations for option pricing
  - Cumulative normal distribution function
  - European call and put option pricing
- Parameters varied:
  - Strike prices (K): 80-120% of spot price
  - Time to maturity (T): 1 day to 1 year
  - Volatility (σ): 10-50%
  - Risk-free rate (r): 0-5%
  - Dividend yield (q): 0-3%

#### 1.2 Monte Carlo Simulation
- Implemented in `src/data/monte_carlo.py`
- Key features:
  - Geometric Brownian Motion for asset price simulation
  - Antithetic variates for variance reduction
  - Control variates for improved accuracy
  - Parallel processing for large simulations
- Error management:
  - Confidence intervals calculation
  - Convergence analysis
  - Variance reduction techniques

### 2. Data Processing Pipeline

#### 2.1 Feature Engineering
- Implemented in `src/features/feature_engineering.py`
- Key transformations:
  - Moneyness (S/K)
  - Time to maturity in years
  - Volatility surface interpolation
  - Interest rate term structure
  - Dividend yield adjustments
- Derived features:
  - Greeks (Delta, Gamma, Vega, Theta, Rho)
  - Implied volatility
  - Historical volatility

#### 2.2 Data Validation
- Implemented in `src/data/validation.py`
- Checks performed:
  - No-arbitrage conditions
  - Put-call parity
  - Option price bounds
  - Greeks consistency

### 3. Model Development Pipeline

#### 3.1 Model Architecture
- Implemented in `src/models/`
- Models included:
  - Linear Regression (baseline)
  - Random Forest
  - XGBoost
  - Neural Networks (MLP and LSTM)
- Feature importance analysis
- Hyperparameter optimization

#### 3.2 Training Process
- Cross-validation strategy:
  - Time-based splits
  - K-fold cross-validation
- Performance metrics:
  - RMSE, MAE
  - R-squared
  - Greeks accuracy
  - Computational efficiency

### 4. Exotic Options Extension

#### 4.1 Worst-Off Option Implementation
- Implemented in `src/models/exotic_options.py`
- Key components:
  - Multi-asset Monte Carlo simulation
  - Correlation structure modeling
  - Path-dependent feature engineering
- Challenges addressed:
  - Dimensionality curse
  - Correlation modeling
  - Computational efficiency

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project is organized into several Jupyter notebooks that guide through the analysis:

1. **Exploratory Data Analysis (EDA)**:
   - `01_EDA_European_Vanilla.ipynb`: Analysis of vanilla European options
   - `04_EDA_Worst_Off.ipynb`: Analysis of Worst-Off options

2. **Data Processing**:
   - `02_Data_Transformation.ipynb`: Data preprocessing and feature engineering

3. **Model Development**:
   - `03_Models.ipynb`: Training and evaluation of machine learning models

## Key Components

### 1. Synthetic Data Generation
- Black-Scholes formula implementation for vanilla options
- Monte Carlo simulation for numerical pricing
- Error analysis and management in Monte Carlo methods

### 2. Machine Learning Pipeline
- Feature engineering and preprocessing
- Model training and evaluation
- Performance metrics and visualization

### 3. Exotic Options Extension
- Implementation of European Worst-Off options
- Comparative analysis with vanilla options
- Discussion of path-dependent derivatives

## Dependencies

The project requires Python 3.x and the following key packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- torch
- xgboost

See `requirements.txt` for the complete list of dependencies with specific versions.

## Results and Visualizations

The project includes various visualizations and analysis results stored in the `results/` directory:
- Payoff surfaces
- Pricing behavior under different market conditions
- Model performance metrics
- Error analysis

## Future Improvements

With additional time and resources, the following enhancements could be implemented:
- More sophisticated feature engineering
- Advanced model architectures
- Real-time market data integration
- Extended support for other exotic options
- Parallel processing for Monte Carlo simulations

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license] 