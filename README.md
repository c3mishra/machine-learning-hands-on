# Machine Learning Hands-On

## Linear Regression: GDP to Life Satisfaction Prediction

This project demonstrates a simple linear regression model that predicts life satisfaction based on GDP per capita using scikit-learn.

### Requirements

- Python >= 3.10
- scikit-learn >= 1.6.1
- matplotlib
- pandas
- numpy
- packaging

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

2. Install dependencies:
```bash
pip install scikit-learn matplotlib pandas numpy packaging
```

### Usage

Run the main script:
```bash
python linear-regression-gdp-life-satisfaction-prediction.py
```

The script will:
- Load life satisfaction data from a remote CSV
- Train a linear regression model
- Predict life satisfaction for a given GDP per capita value

### Data Source

Data is loaded from: `https://github.com/ageron/data/raw/main/lifesat/lifesat.csv`

### Features

- Data visualization (commented out by default)
- Model training using scikit-learn's LinearRegression
- Prediction for new GDP values
