# Logistic Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def logreg_classify(file_name):

    df = pd.read_csv(file_name, delimiter='\t', quoting=3)
    y = df.iloc[:, -1].values
    # Cleaning Text

    # Contains all different reviews as clean data
    corpus = []

    for iterator in range(0, 1000):
        # Replacing any non letter like commas, dots, etc. with spaces using regular expressions
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][iterator])

        # Transforming all letters to lower case
        review = review.lower()

        # Splitting each review to a list of words
        review = review.split()

        # - Applying stemming technique to reduce the complexity by simplifying each word to its root (e.g, refreshing ==>
        # refresh, loved ==> love, etc.) using porter stemmer which is a classical type of stemmer
        port_stem = PorterStemmer()

        # Removing not word from stopwords because it always gives us a negative feedback if exists so it'll improve
        # our model if existed in the review
        all_stop_words = stopwords.words('english')
        all_stop_words.remove('not')

        # - Removing also stop words (e.g, she, is , a, etc.) to reduce complexity
        review = [port_stem.stem(word=word) for word in review if not word in set(all_stop_words)]
        # Reconstructing the review as a text by join method that joins all list elements as a single string seperated by space
        review = ' '.join(review)
        corpus.append(review)

    # Tokenization: Converting a collection of text documents to a matrix of token counts producing a sparse
    # representation of the counts

    # max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency
    # across the corpus
    count_vect = CountVectorizer(max_features=1500)
    X = count_vect.fit_transform(corpus).toarray()

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


    # Applying Grid Search
    # parameters = [{'C': [0.25, 0.5, 0.75, 1], 'penalty': ['l2'], 'solver':
    #               ['lbfgs','newton-cg','sag'], 'multi_class': ['auto']},
    #               {'C': [0.25, 0.5, 0.75, 1], 'penalty': ['l1'], 'solver':['liblinear', 'saga'],
    #                 'multi_class': ['auto']}]
    # grid_search = GridSearchCV(estimator=LogisticRegression(random_state=0),
    #                            param_grid=parameters,
    #                            scoring='accuracy',
    #                            cv=10,
    #                            n_jobs=-1)
    #
    # grid_search.fit(X_train,y_train)
    # best_params = grid_search.best_params_
    # print("Best Accuracy: {:.2f} %".format(grid_search.best_score_ * 100))
    # print("Best Parameters:", best_params)
    #
    # log_classifier = LogisticRegression(random_state=0, C=best_params['C'], penalty=best_params['penalty'],
    #                                     solver= best_params['solver'], max_iter=best_params['max_iter'], multi_class='auto')

    log_classifier = LogisticRegression(random_state=0)
    log_classifier.fit(X_train,y_train)
    y_pred = log_classifier.predict(X_test)
    print('Logistic Regression')
    print('Accuracy: ', accuracy_score(y_test,y_pred)*100)
    print('Precision: ', precision_score(y_test,y_pred)*100)
    print('Recall: ', recall_score(y_test,y_pred)*100)
    print('F1 Score: ', f1_score(y_test,y_pred)*100)

