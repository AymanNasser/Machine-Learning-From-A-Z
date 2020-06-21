import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Loading data set
df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

X = df.iloc[:, :-1].values # Features
y = df.iloc[:, -1].values # Label

# Encoding Gender column using 1-Hot encoding
label_encode = LabelEncoder()
X[:,2] = label_encode.fit_transform(X[:,2])

# Encoding Geography column using 1-Hot encoding
col_transform = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [1])], remainder= 'passthrough')
X = np.array(col_transform.fit_transform(X))

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

# Applying feature scaling to all features *Import. for deep learning model*
stand_scaler = StandardScaler()
X_train = stand_scaler.fit_transform(X_train)
X_test = stand_scaler.transform(X_test)

# Building ANN as a sequence of layers
ann = tf.keras.models.Sequential()
# Adding the input & 1st hidden layer
# uints=6 : 1st hidden layer has 6 nodes with activation function = 'relu': rectifier
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding output layer
# units = 1: because our classification model is binary class.
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training ANN
ann.compile(metrics=['accuracy','Precision'], optimizer='adam', loss='binary_crossentropy')
ann.fit(X_train,y_train, batch_size = 32, epochs = 50)

# Predicting a new example
# France: 1,0,0. Female: 0
print_str = 'Probability if the customer is leaving the bank'
print(print_str,ann.predict(stand_scaler.transform( [[1,0,0,600,1,40,3,60000,2,1,1,50000]] ) ) )

# Evaluating ANN
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print('Accuracy: ', accuracy_score(y_test,y_pred))
print('F1 Score: ', f1_score(y_test,y_pred))
