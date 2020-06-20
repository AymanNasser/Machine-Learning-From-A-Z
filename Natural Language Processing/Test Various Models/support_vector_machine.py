# Support Vector Machine (SVM)

# Importing the libraries
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

def lsvm_classify(file_name):
    df = pd.read_csv(file_name, delimiter='\t', quoting=3)
    y = df.iloc[:, -1].values

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

    # Training the SVM model on the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)

    new_review = 'Worst restaurant ever'
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = count_vect.transform(new_corpus).toarray()
    new_y_pred = classifier.predict(new_X_test)
    print(new_y_pred)

    print('Linear SVM')
    print('Accuracy: ', accuracy_score(y_test,y_pred)*100)
    print('Precision: ', precision_score(y_test,y_pred)*100)
    print('Recall: ', recall_score(y_test,y_pred)*100)
    print('F1 Score: ', f1_score(y_test,y_pred)*100)