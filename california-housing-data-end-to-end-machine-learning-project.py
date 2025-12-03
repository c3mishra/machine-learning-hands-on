import pandas as pd
from pathlib import Path
import urllib.request
import tarfile

def download_housing_data():
    housing_tarfile_path = Path("datasets/housing.tgz")
    if not housing_tarfile_path.is_file():
        Path("datasets").mkdir(parents = True, exist_ok = True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"

        urllib.request.urlretrieve(url, housing_tarfile_path)

        with tarfile.open(housing_tarfile_path) as housing:
            housing.extractall(path = "datasets", filter = "data")
    return pd.read_csv("datasets/housing/housing.csv")

housing_full_data = download_housing_data()
print(housing_full_data.head())

print(housing_full_data.info())

print(housing_full_data.describe())

print(housing_full_data["ocean_proximity"].value_counts())

# lets visualize some attributes
import matplotlib.pyplot as plt

# housing_full_data.hist(bins = 50, figsize = (14, 12))
# plt.show()

# lets split data into training and test sets using custom method

import numpy as np

# def split(data, testratio, rng):
#     shuffled_indices = rng.permutation(len(data))
#     test_set_size = (int)(len(data) * testratio)

#     test_set_indices = shuffled_indices[:test_set_size]
#     train_set_indices = shuffled_indices[test_set_size:]

#     return data.iloc[train_set_indices], data.iloc[test_set_indices]

# rng = np.random.default_rng()
# train_set, test_set = split(housing_full_data, 0.2, rng)

# print(f"Train set size: {len(train_set)}")
# print(f"Test set size: {len(test_set)}")

# split using standard method from sklearn
from sklearn.model_selection import train_test_split

train_set_random, test_set_random = train_test_split(housing_full_data, test_size = 0.2, random_state = 42)

# split using stratify based on median_income

# lets categarorize data based on median_income
housing_full_data["income_cat"] = pd.cut(housing_full_data["median_income"], bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels = [1, 2, 3, 4, 5])
train_set_stratify, test_set_stratify = train_test_split(housing_full_data, test_size = 0.2, random_state = 42, stratify = housing_full_data["income_cat"])

# lets drop income_cat column now as its no longer required

for set_ in train_set_stratify, test_set_stratify:
    set_.drop("income_cat", axis = 1, inplace = True)

# now lets analyze train data and find standard correlation coefficient or pearson's r
housing = train_set_stratify.copy()

corr_matrix = housing.corr(numeric_only = True)
print(corr_matrix["median_house_value"].sort_values(ascending = False))

# now lets clean and trasform data
# there are below trasformations we will perform
# fix missing values in numeric features
# convert non numeric values to numeric

# lets split housing and housing_labels first
housing = train_set_stratify.drop("median_house_value", axis = 1)
housing_labels = train_set_stratify["median_house_value"].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median") 
# startegy can be median, mean, most_frequent or (constant, fill_value=<value>)
housing_num = housing.select_dtypes(include = [np.number])

imputer.fit(housing_num)

X = imputer.transform(housing_num)

housing_imputed = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
print(housing_imputed.head())

# use one hot encodig to convert text data to numeric

from sklearn.preprocessing import OneHotEncoder
# there is also OrdinalEncoder which assigns numeric values to text data
housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder(sparse_output = False)
housing_cat_onehot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_onehot)

# full piepline for data cleaning and trasformation

from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

housing_num_pipeline = make_pipeline(SimpleImputer(strategy = "median"), StandardScaler())
housing_cat_pipeline = make_pipeline(SimpleImputer(strategy = "most_frequent"), OneHotEncoder(sparse_output = False))

# The startegy to fill missing data can be median, mean, most_frequent, constant - fill_value
# other types of imputer - KNNImputer, IterativeImputer
# other types of scaler MinMaxScaler

housing_num_column_selector = make_column_selector(dtype_include = np.number)
housing_cat_column_selector = make_column_selector(dtype_include = object)

preprocessing = make_column_transformer(
    (housing_num_pipeline, housing_num_column_selector),
    (housing_cat_pipeline, housing_cat_column_selector)
)

housing_preprocessed = preprocessing.fit_transform(housing)
housing_df = pd.DataFrame(housing_preprocessed, columns = preprocessing.get_feature_names_out())
print(housing_df.info())