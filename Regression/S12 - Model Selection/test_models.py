from decision_tree_regression import dt_reg
from multiple_linear_regression import ml_reg
from polynomial_regression import pl_reg
from random_forest_regression import rf_reg
from support_vector_regression import sv_reg

print('All models are tested without parameters tuning ')
print('R2 of Multiple Linear Regression ',ml_reg('Data.csv'))
print('R2 of Polynomial Regression ',pl_reg('Data.csv'))
print('R2 of Support Vector Regression ',sv_reg('Data.csv'))
print('R2 of Decision Tree ',dt_reg('Data.csv'))
print('R2 of Random Forrest Regression ',rf_reg('Data.csv'))

