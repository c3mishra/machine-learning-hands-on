# Machine Learning Hands-On

A collection of machine learning projects demonstrating various algorithms and techniques using scikit-learn.

## Projects

### 1. GDP to Life Satisfaction Prediction

This project demonstrates machine learning regression models that predict life satisfaction based on GDP per capita using scikit-learn. Two different algorithms are implemented:

1. **Linear Regression** - A simple linear model
2. **K-Nearest Neighbors (KNN) Regressor** - A non-parametric algorithm using 3 neighbors

### 2. California Housing Price Prediction (End-to-End ML Project)

An end-to-end machine learning project that predicts median house prices in California using the California Housing dataset. This comprehensive project demonstrates:

- **Data acquisition and loading** from remote sources
- **Exploratory Data Analysis (EDA)** with correlation analysis
- **Data splitting strategies**:
  - Random splitting
  - Stratified splitting based on income categories
- **Data preprocessing pipeline**:
  - Handling missing values with SimpleImputer
  - Numerical feature scaling with StandardScaler
  - Categorical encoding with OneHotEncoder
  - Automated column selection and transformation
- **Model training and evaluation**:
  - Linear Regression with RMSE evaluation
  - Decision Tree Regressor
  - Random Forest Regressor
  - Cross-validation with 10-fold CV for robust performance assessment
  - Model comparison using RMSE metrics

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

**Life Satisfaction Models:**

Run the Linear Regression model:
```bash
python linear-regression-gdp-life-satisfaction-prediction.py
```

Run the K-Nearest Neighbors model:
```bash
python k-nearest-neighbors-regressor-life-satisfaction-prediction.py
```

**California Housing Price Prediction:**

```bash
python california-housing-data-end-to-end-machine-learning-project.py
```

Both life satisfaction scripts will:
- Load life satisfaction data from a remote CSV
- Train their respective model
- Predict life satisfaction for a given GDP per capita value

### Data Sources

- **Life Satisfaction Dataset**: `https://github.com/ageron/data/raw/main/lifesat/lifesat.csv`
- **California Housing Dataset**: `https://github.com/ageron/data/raw/main/housing.tgz`

### Features

- Data visualization and exploratory data analysis
- **Linear Regression**: Simple linear model for prediction
- **K-Nearest Neighbors**: Non-parametric regression using 3 nearest neighbors
- **Decision Tree Regressor**: Non-linear regression using decision trees
- **Random Forest Regressor**: Ensemble method combining multiple decision trees
- **End-to-End ML Pipeline**: Complete workflow from data acquisition to model evaluation
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding
- **Cross-Validation**: 10-fold cross-validation for robust model assessment
- **Model Comparison**: RMSE-based evaluation across multiple algorithms
- Comparison of different ML approaches on multiple datasets
