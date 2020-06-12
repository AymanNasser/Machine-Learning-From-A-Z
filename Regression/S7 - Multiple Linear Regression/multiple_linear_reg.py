from data_preprocessing_template import data_preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train , y_test = data_preprocessing('50_Startups.csv')

print(X_train)

