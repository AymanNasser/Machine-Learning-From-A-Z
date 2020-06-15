# Classification Notes

## Logistic Regression
- Logistic regression is linear classifier 
- Logistic regression returns probabilities

> Confusion matrix: A confusion matrix is a table that is often used to describe the **performance** 
> of a classification model (or "classifier") on a set of test data for which the true values are known.
>
> Let's now define the most basic terms, which are whole numbers (not rates):
> - true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
> - true negatives (TN): We predicted no, and they don't have the disease.
> - false positives (FP): We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
> - false negatives (FN): We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
> 
> his is a list of rates that are often computed from a confusion matrix for a binary classifier:
> ~~~~
> Accuracy: Overall, how often is the classifier correct? `(TP+TN)/total`
>
> Mis-classification Rate: Overall, how often is it wrong? `(FP+FN)/total`,
> equivalent to 1 minus Accuracy also known as "Error Rate"
>
> True Positive Rate: When it's actually yes, how often does it predict yes? `TP/actual yes`  
> also known as "Sensitivity" or "Recall"
>
> False Positive Rate: When it's actually no, how often does it predict yes? `FP/actual no`
>
> True Negative Rate: When it's actually no, how often does it predict no? `TN/actual no`
>
> Precision: When it predicts yes, how often is it correct? `TP/predicted yes`
> 
> Prevalence: How often does the yes condition actually occur in our sample? `actual yes/total`
 ~~~~
