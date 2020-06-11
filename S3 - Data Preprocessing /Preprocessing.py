# Importing the libraries
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

# Importing the dataset & creating data frame
dataset = pd.read_csv('Data.csv')

# iloc ==> locate indexes [rows, cols]
# indexes[:, :-1] ==> Taking all rows [:] & taking all columns excluding the last one [:-1]
X = dataset.iloc[:, :-1].values # Features
y = dataset.iloc[:, -1].values # Label

# Taking care of missing data
# Replacing empty values by specifying working on cell which (missing_values=np.nan)
# Replacing these empty values by mean values of each feature
imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')
# Selecting all numerical columns to replace all missing data. In our case they're ages & salary features
imputer.fit(X[:, 1:3])
# Replacing using transform
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)

# Encoding categorical data using 1-Hot encoding
# We encode categorical data to prevent our model from mis-interpreting some co-relations between features & labels

# ColumnTransformer(transformers, *, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False)
# transformers list of tuples : List of (name of transformation, transformer, columns to be transformed) tuples
# specifying the transformer objects to be applied to subsets of the data.

# Remainder ==> keep the columns that wont apply to the transformation
col_transform = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')
X = np.array(col_transform.fit_transform(X))
#print(X)

# Encoding labels
label_encode = LabelEncoder()
y = label_encode.fit_transform(y)
#print(y)

# Splitting the data set into the Training set and Test set
# random_state = 1 ==> fixing the randomization to always produces the same randomized output
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 1)

# Feature scaling
stand_scaler = StandardScaler()
# We're indexing columns starting from 3 to the last as when we encoded the categorical data we split
# Country feature to 3 columns so Age feature index = 3
X_train[:, 3:] = stand_scaler.fit_transform(X_train[:, 3:])
# Applying the same scaler that applied to the training set
X_test[:, 3:] = stand_scaler.transform(X_test[:, 3:])

