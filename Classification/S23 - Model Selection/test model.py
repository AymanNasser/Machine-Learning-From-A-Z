from decision_tree_classification import dt_classify
from kernel_svm import ksvm_classify
from logistic_regression import logreg_classify
from naive_bayes import nb_classify
from k_nearest_neighbors import knn_classify
from random_forest_classification import rf_classify
from support_vector_machine import lsvm_classify

print('Decision Tree Model: ', dt_classify('breast_cancer.csv'))
print('Random Forrest Model: ', rf_classify('breast_cancer.csv'))
print('K-NN Model: ', knn_classify('breast_cancer.csv'))
print('Linear SVM Model: ', lsvm_classify('breast_cancer.csv'))
print('Kernel SVM Model: ', ksvm_classify('breast_cancer.csv'))
print('Logistic Regression Model: ', logreg_classify('breast_cancer.csv'))
print('Naive Bayes Model: ', nb_classify('breast_cancer.csv'))

