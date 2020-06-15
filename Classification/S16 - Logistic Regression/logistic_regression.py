from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Loading data
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size= 0.20)

# Feature scaling isn't necessary to be implemented
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test =  std_scaler.transform(X_test)


# Creating logistic regressor
log_regressor = LogisticRegression(random_state=0, max_iter=100, multi_class='ovr')
log_regressor.fit(X_train,y_train)
y_predicted = log_regressor.predict(X_test)


#print(log_regressor.predict(std_scaler.transform([[30,87000]])))
print('Confusion Matrix: \n',confusion_matrix(y_test,y_predicted))
print('Accuracy',accuracy_score(y_test,y_predicted))



