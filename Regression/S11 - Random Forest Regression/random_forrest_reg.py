from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values # Features
y = dataset.iloc[:, -1].values # Label

# Creating random regressor
# Parameters:
# - n_estimators int, default=100 , The number of trees in the forest
# - criterion{“mse”, “mae”}, default=”mse” The function to measure the quality of a split. Supported criteria are
# “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for
# the mean absolute error.
# - max_depth int, default=None, The maximum depth of the tree. If None, then nodes are expanded until all leaves are
# pure or until all leaves contain less than min_samples_split samples.
# - min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
# min_impurity_decrease=0.0, min_impurity_split=None, etc

rand_regressor = RandomForestRegressor(n_estimators=500, max_depth=None)
rand_regressor.fit(X,y)

print(rand_regressor.predict([[6.5]]))

# Visualizing random foresst regressor wit high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, rand_regressor.predict(X_grid), color = 'blue')
plt.title('Salary vs Experience (Random Forrest Regressor)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()