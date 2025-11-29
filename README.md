# Machine Learning Hands-On

## GDP to Life Satisfaction Prediction

This project demonstrates machine learning regression models that predict life satisfaction based on GDP per capita using scikit-learn. Two different algorithms are implemented:

1. **Linear Regression** - A simple linear model
2. **K-Nearest Neighbors (KNN) Regressor** - A non-parametric algorithm using 3 neighbors

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

Run the Linear Regression model:
```bash
python linear-regression-gdp-life-satisfaction-prediction.py
```

Run the K-Nearest Neighbors model:
```bash
python k-nearest-neighbors-regressor-life-satisfaction-prediction.py
```

Both scripts will:
- Load life satisfaction data from a remote CSV
- Train their respective model
- Predict life satisfaction for a given GDP per capita value

### Data Source

Data is loaded from: `https://github.com/ageron/data/raw/main/lifesat/lifesat.csv`

### Features

- Data visualization (commented out by default)
- **Linear Regression**: Simple linear model for prediction
- **K-Nearest Neighbors**: Non-parametric regression using 3 nearest neighbors
- Comparison of different ML approaches on the same dataset
