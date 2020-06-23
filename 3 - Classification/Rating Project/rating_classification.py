import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
# Loading data
df_train = pd.read_csv('sample_train.csv')
df_test = pd.read_csv('sample_dev.csv')

# Dropping un necessary columns. axis = 1 for deleting column
df_train = df_train.drop('reviews.text',axis=1)
df_test = df_test.drop('reviews.text',axis=1)

X_train = df_train.iloc[:, :-1].values
X_test = df_test.iloc[:, :-1].values

y_train = df_train.iloc[:, -1].values
y_test = df_test.iloc[:, -1].values

# Encoding labels
label_encode = LabelEncoder()
y_train = label_encode.fit_transform(y_train)
y_test = label_encode.transform(y_test)

# Creating a decision tree classifier object
tree_classifier = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=7, min_samples_split=2)
tree_classifier.fit(X_train,y_train)
y_predicted = tree_classifier.predict(X_test)

conf_mat = confusion_matrix(y_test,y_predicted)
print('Decision Tree')
print('True Positives', conf_mat[0,0], 'False Positives', conf_mat[0,1])
print('False Negatives', conf_mat[1,0], 'True Negatives', conf_mat[1,1])
print('Accuracy', accuracy_score(y_test,y_predicted))

# Creating a naive bayes classifier object
naiveBayes_calssifier = CategoricalNB()
naiveBayes_calssifier.fit(X_train,y_train)
y_predicted = naiveBayes_calssifier.predict(X_test)

conf_mat = confusion_matrix(y_test,y_predicted)
print('\nNaive Bayes')
print('True Positives', conf_mat[0,0], 'False Positives', conf_mat[0,1])
print('False Negatives', conf_mat[1,0], 'True Negatives', conf_mat[1,1])
print('Accuracy', accuracy_score(y_test,y_predicted))

# Creating a random forrest classifier
randomForrest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth=5, max_features='auto',random_state = 0)
randomForrest_classifier.fit(X_train,y_train)
y_predicted = randomForrest_classifier.predict(X_test)

conf_mat = confusion_matrix(y_test,y_predicted)
print('\nRandom Forrest')
print('True Positives', conf_mat[0,0], 'False Positives', conf_mat[0,1])
print('False Negatives', conf_mat[1,0], 'True Negatives', conf_mat[1,1])
print('Accuracy', accuracy_score(y_test,y_predicted))

# Creating a kernel SVM classifier
supportVector_classifier = SVC(kernel='rbf')
supportVector_classifier.fit(X_train,y_train)
y_predicted = supportVector_classifier.predict(X_test)

conf_mat = confusion_matrix(y_test,y_predicted)
print('\nNon Linear Support Vector')
print('True Positives', conf_mat[0,0], 'False Positives', conf_mat[0,1])
print('False Negatives', conf_mat[1,0], 'True Negatives', conf_mat[1,1])
print('Accuracy', accuracy_score(y_test,y_predicted))

