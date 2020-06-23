import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_excel('Power_Plant.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

# Applying feature scaling to all features *Import. for deep learning model*
stand_scaler = StandardScaler()
X_train = stand_scaler.fit_transform(X_train)
X_test = stand_scaler.transform(X_test)

# Building An ANN
ann = tf.keras.models.Sequential()

# Adding 2 hidden layers with 6 nodes
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding output layer
ann.add(tf.keras.layers.Dense(units=1))

# Training ANN
ann.compile(metrics=[tf.metrics.RootMeanSquaredError()],optimizer='adam', loss='mean_squared_error')
ann.fit(X_train,y_train, batch_size = 32, epochs = 50)

