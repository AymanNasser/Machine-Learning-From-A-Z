# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Decision tree
# criterion{“gini”, “entropy”}, default=”gini” The function to measure the quality of a split. Supported criteria
# are “gini” for the Gini impurity and “entropy” for the information gain.
tree_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=0, criterion= 'entropy')
tree_classifier.fit(X_train,y_train)
y_pred = tree_classifier.predict(X_test)

# Predicting a new result
print(tree_classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = tree_classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
