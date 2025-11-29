import sys

# check pythin version
assert sys.version_info >= (3,10)

# check scikit learn version
from packaging.version import Version
import sklearn
assert Version(sklearn.__version__) >= Version("1.6.1")

import matplotlib.pyplot as plt
plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
plt.rc('font', size = 12)
plt.rc('axes', labelsize = 14, titlesize = 14)
plt.rc('legend', fontsize = 12)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

countries = lifesat["Country"].values
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# uncomment to checkout the data
# print(countries)
# print(X)
# print(y)

# uncomment to visualize the data
# lifesat.plot(kind='scatter', grid = True, x='GDP per capita (USD)', y = 'Life satisfaction')
# plt.axis([25000, 65000, 4, 9])
# plt.show()

# train the model
model = LinearRegression()
model.fit(X, y)

new_X = [[33442.8]]
print(model.predict(new_X))