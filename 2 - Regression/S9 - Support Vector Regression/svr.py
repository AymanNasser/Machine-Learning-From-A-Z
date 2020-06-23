from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values # Features
y = dataset.iloc[:, -1].values # Label

# Feature scaling

# Transforming y to 2d array as standard scaler class expects 2d array as input
# reshape(no. of rows, no. of cols)
y = y.reshape(len(y),1)

# Applying std scaler for X to calc. the mean & standard deviation for X only
std_scaler_X = StandardScaler()
X = std_scaler_X.fit_transform(X)

std_scaler_Y = StandardScaler()
y = std_scaler_Y.fit_transform(y)

# Training the model
svr_regressor = SVR(kernel= 'rbf')
svr_regressor.fit(X,y.ravel())

# Reverse scaling
# Predicting a specific value for x after applying the same feature scaling applied for X then inverse
# transform the scale to predict y in its original scale
print( std_scaler_Y.inverse_transform(svr_regressor.predict(std_scaler_X.transform( [[6.5]] ))) )

# Visualizing SVR
plt.scatter(std_scaler_X.inverse_transform(X),std_scaler_Y.inverse_transform(y),color = 'red')
plt.plot(std_scaler_X.inverse_transform(X), std_scaler_Y.inverse_transform(svr_regressor.predict(X)), color = 'blue')
plt.title('Salary vs Experience (Support Vector Regression)')
plt.xlabel('Experience')
plt.ylabel('Salary')


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(std_scaler_X.inverse_transform(X)), max(std_scaler_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(std_scaler_X.inverse_transform(X), std_scaler_Y.inverse_transform(y), color = 'yellow')
plt.plot(X_grid, std_scaler_Y.inverse_transform(svr_regressor.predict(std_scaler_X.transform(X_grid))), color = 'gray')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




