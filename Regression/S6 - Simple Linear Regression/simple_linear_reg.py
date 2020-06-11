from data_preprocessing_template import data_preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train , y_test = data_preprocessing('Salary_Data.csv')

# Creating an object of linear regression without specifying parameters as it is a simple LR
regressor = LinearRegression()

# Training our data using fit method
regressor.fit(X_train,y_train)

ytest_predicted = regressor.predict(X_test)
#print(abs(y_test-ytest_predicted))

# Visualizing training sets
plt.scatter(X_train,y_train,color = 'red')
# plt.plot(): Plots y versus x as lines and/or markers.
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (TRAINING)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing test sets
plt.scatter(X_test,y_test,color = 'yellow')
# Plotting our output of linear model that trained on X_train
ytrain_predicted = regressor.predict(X_train)
plt.plot(X_train, ytrain_predicted , color = 'black')
plt.title('Salary vs Experience (TEST)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


