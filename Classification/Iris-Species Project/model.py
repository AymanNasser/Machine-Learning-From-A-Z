import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
# Loading data set
data_set = pd.read_csv('Iris.csv')
data_set = data_set.drop('Id', axis=1)

X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

# Encoding output labels
label_encode = LabelEncoder()
y = label_encode.fit_transform(y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

# Feature scaling
stand_scaler = StandardScaler()
X_train = stand_scaler.fit_transform(X_train)
X_test = stand_scaler.transform(X_test)

# Training set on various classification models

# Support Vector Machine Model
svm_classifier = SVC(kernel='rbf', random_state=0)
svm_classifier.fit(X_train,y_train)
y_predicted = svm_classifier.predict(X_test)

print('SVM')
print('Accuracy: ', accuracy_score(y_test,y_predicted))

# K-NN Model
k_nn = KNeighborsClassifier(n_neighbors=5, p=2, algorithm='auto', metric= 'minkowski')
k_nn.fit(X_train,y_train)
y_predicted = k_nn.predict(X_test)

print('K-NN')
print('Accuracy',accuracy_score(y_test,y_predicted))


