from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Loading data
dataset = pd.read_csv('../Social_Network_Ads.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size= 0.20)

# Feature scaling isn't necessary to be implemented
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test =  std_scaler.transform(X_test)


# Creating K-NN model
# - n_neighborsint, default=5, Number of neighbors to use by default for k-neighbors queries
# - p int, default = 2, Power parameter for the Minkowski metric. When p = 1, this is equivalent to using
# manhattan_distance (l1) and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
k_nn = KNeighborsClassifier(n_neighbors=5, p=2, algorithm='auto', metric= 'minkowski')
k_nn.fit(X_train,y_train)

y_predicted = k_nn.predict(X_test)

print(k_nn.predict(std_scaler.transform([[30,87000]])))
print('Confusion Matrix: \n',confusion_matrix(y_test,y_predicted))
print('Accuracy',accuracy_score(y_test,y_predicted))






