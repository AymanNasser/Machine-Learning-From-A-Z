from decision_tree_classification import dt_classify
from kernel_svm import ksvm_classify
from logistic_regression import logreg_classify
from naive_bayes import nb_classify
from k_nearest_neighbors import knn_classify
from random_forest_classification import rf_classify
from support_vector_machine import lsvm_classify

logreg_classify('../Restaurant_Reviews.tsv')
print('********************************************')
knn_classify('../Restaurant_Reviews.tsv')
print('********************************************')
dt_classify('../Restaurant_Reviews.tsv')
print('********************************************')
ksvm_classify('../Restaurant_Reviews.tsv')
print('********************************************')
lsvm_classify('../Restaurant_Reviews.tsv')
print('********************************************')
rf_classify('../Restaurant_Reviews.tsv')


