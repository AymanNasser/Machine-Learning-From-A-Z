# Data Preprocessing Template

# Importing the libraries
import pandas as pd

def data_preprocessing(file_name):
    # Importing the dataset
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test