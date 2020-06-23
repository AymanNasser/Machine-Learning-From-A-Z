from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values # Features
y = dataset.iloc[:, -1].values # Label


col_transform = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [3])], remainder= 'passthrough')
X = np.array(col_transform.fit_transform(X))

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training model
# Don't worry about either dummy variable trap or selecting features with the highest P-Value as Sklearn class
# will take care of this
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_predicted = regressor.predict(X_test)
# Printing y_predicted & y_actual for test data set as a 2d vector for better visualizations
np.set_printoptions(precision=2)
print(np.concatenate( (y_predicted.reshape(len(y_predicted),1), y_test.reshape(len(y_test),1)), 1))








