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
