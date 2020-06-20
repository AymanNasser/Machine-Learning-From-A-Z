import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading data
# delimiter = '\t' to specify the read function that we're reading tab sep. var file
# quoting= 3 to ignore the double quotes of text in the tsv file
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting= 3)
y = df.iloc[:,-1].values
# Cleaning Text
nltk.download('stopwords')

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
    review = [port_stem.stem(word= word) for word in review if not word in set(all_stop_words) ]
    # Reconstructing the review as a text by join method that joins all list elements as a single string seperated by space
    review = ' '.join(review)
    corpus.append(review)

# Tokenization: Converting a collection of text documents to a matrix of token counts producing a sparse
# representation of the counts

# max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency
# across the corpus
count_vect = CountVectorizer(max_features= 1500)
X = count_vect.fit_transform(corpus).toarray()
print('Number of words resulted from tokenization = ',len(X[0]))

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.20, random_state= 0)

# Training naive bayes model
gauss_nb = GaussianNB()
gauss_nb.fit(X_train,y_train)

y_predict = gauss_nb.predict(X_test)
print('Confusion Matrix: ',confusion_matrix(y_test,y_predict))
print('Model Accuracy: ', accuracy_score(y_test,y_predict))

accuracies = cross_val_score(estimator= gauss_nb, X=X_train, y=y_train, cv= 10)
print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))


