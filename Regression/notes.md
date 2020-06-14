# Regression Notes

Assumptions of Linear Regression:
1. Linearity
2. Homoscedasticity
3. Multivariate normality
4. Independence of errors
5. Lack of multicollinearity
> The linearity in regressions refers to the coefficients not the variables 
   
> For every ***categorical variable*** in a specific column we usually set each variable 
in a seperate column working as switches (***dummy variable*** ) 
>
> Always __omit (remove)__ one dummy variable from model irrespective of no. of dummy 
> variables 

> Statistical significance: it's a point when in human intuitive terms, you get uneasy about the 
> null hypothesis being true

> Ensemble learning: is when we take **multiple** algorithms or **same** algorithm multiple times
> and put them together to make something much more powerful than the original 
*** 

### Simple LR
- To retrieve a specific value for **input(x)** we use `regressor.predict([[x]])`
- To retrieve theta_0 & theta_1 __coefficients__ we use `regressor.coef_` & `regressor.intercept`
as our `f(x) = intercept_ + coef_ * x`

***
### Multiple LR
In Multiple LR we don't need to do ***feature scaling*** due to multiplied coff. (b0,b1,etc)

### Methods of Building a Model
Multiple Linear Regression has several techniques to build an effective model namely:

1. All-in 
2. Backward elimination 
3. Forward selection
4. Bidirectional elimination

#### All-in
Throwing all our variables to the model

#### Backward elimination 
Steps:
1. Select a significance level to stay in the model (eg. `SL = 0.05`)
2. Fit the model with all possible predictors
3. Consider the predictor with the highest P-value. If `P>SL`, this predictor is excluded
4. Remove (**eliminate**) the predictor
5. Fit the model without this variable and repeat the step c until the condition `P>SL` 
becomes false 
> At the end, we'll have variables with P-Value less than Selected SL

#### Forward selection
Steps:
1. Select a significance level to enter in the model (eg. SL = 0.05)
2. Fit all simple regression models y ~ Xn **Select** the one with the lowest P-Value
3. Keep this variable and keep fit all possible models with one extra predictor added to the 
one(s) we already have 
4. Consider the predictor with the lowest P-Value. If P < SL, go to step-3, otherwise model is finished

#### Bidirectional elimination
A combination of the above, testing at each step for variables to be included or excluded

***
### Support Vector Regression
Epsilon-intensive tube is a margin of error that we allowed our model to have & not caring 
about any error inside the tube 

### Random Forest Regression
Steps:
1. Pick at random K data points from the training set
2. Build a decision tree associated to these K data points
3. Choose the number of trees we want to build & **repeat** steps 1,2
4. For a new data point, make **each one** of our built trees predict the value of new data 
input & assign the new data point the ***Average across all the predicted Y values***

## Evaluating Regression Models Performance
> R-squared (R2) is a statistical measure that represents the proportion of the variance for 
> a dependent variable that's explained by an independent variable or variables in a regression model. 
> 
> Whereas correlation explains the strength of the relationship between an independent and dependent variable, R-squared explains to what extent the variance of one variable explains the variance of the second variable. So, if the R2 of a model is 0.50, 
> then approximately half of the observed variation can be explained by the model's inputs.
> 
> Sum of squares of residuals (SSres): `SUM(predicted_y(i) - actual_y(i))^2`
> 
> Total sum of squares (SStot): `SUM(predicted_y(i) - average_y(i))^2` 
>
> __`R-squared (R2) = 1 - SSres / SStot`__
> 
> The close R2 gets to **one** the better because, SSres will then equal to zero 
>
> Adjusted (R2): is a good metric as it helps us to understand whether adding good variables
> to model or they're redundant, if we're adding a variable that isn't good for our model
> then, the **(R2) will insignificantly inc.** & the term **P will also inc.** which leads to 
> **dec. of adjusted (R2)**
>
> __`Adjusted (R2) = 1 - (1-(R2))*( n-1 / n-p-1 )  `__ where P: number of independent variables
> and n: sample size

