from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values # Features
y = dataset.iloc[:, -1].values # Label

# We're not Splitting bec. the dataset is too small
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = X
y_train = y

# Training model
# Creating linear model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train,y_train)

# Creating poly regressor, as we inc. the degree we inc. the over-fitting
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X_train)
# Creating new linear regressor
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y_train)

# Visualizing linear reg
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train, linear_regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience Linear')
plt.xlabel('Experience')
plt.ylabel('Salary')
#plt.show()

# Visualizing polynomial reg
plt.scatter(X_train,y_train,color = 'blue')
plt.plot(X_train, lin_reg.predict(X_poly), color = 'yellow')
plt.title('Salary vs Experience Polynomial')
plt.xlabel('Experience')
plt.ylabel('Salary')
#plt.show()

print(linear_regressor.predict([[6.5]]) )
print(lin_reg.predict(poly_regressor.fit_transform([[6.5]])))