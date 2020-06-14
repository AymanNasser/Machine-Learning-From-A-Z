from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values # Features
y = dataset.iloc[:, -1].values # Label


# Creating tree object
tree_regressor = tree.DecisionTreeRegressor(random_state= 0, max_depth=5)
tree_regressor.fit(X,y)

print(tree_regressor.predict([[6.5]]))

# Visualizing tree regressor wit high resolution
# The curve isn't continuous, jumping from position level to another
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, tree_regressor.predict(X_grid), color = 'blue')
plt.title('Salary vs Experience Dec Tree')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
