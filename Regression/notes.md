# Regression Notes

Assumptions of Linear Regression:
1. Linearity
2. Homoscedasticity
3. Multivariate normality
4. Independence of errors
5. Lack of multicollinearity

Always __omit (remove)__ one dummy variable from model


### Simple LR
- To retrieve a specific value for **input(x)** we use `regressor.predict([[x]])`
- To retrieve theta_0 & theta_1 __coefficients__ we use `regressor.coef_` & `regressor.intercept`
as our `f(x) = intercept_ + coef_ * x`

### Multiple LR
 