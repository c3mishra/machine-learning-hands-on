import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

countries = lifesat["Country"].values
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# uncomment to visualize the data
# lifesat.plot(kind='scatter', grid = True, x = 'GDP per capita (USD)', y = 'Life satisfaction')
# plt.axis([25000, 65000, 4, 9])
# plt.show()

model = KNeighborsRegressor(n_neighbors = 3)
model.fit(X, y)

new_X = [[33442.8]]
print(model.predict(new_X))