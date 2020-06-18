# Association Rules
Rules do not extract an individual’s preference, rather find relationships between set of elements of every distinct 
transaction. This is what makes them different from collaborative filtering.

Various metrics are used to help us to understand the **strength** of association between sets

![AssocRule Image](../Images/assocRule.png)

**Support:** 

This measure gives an idea of how **frequent** an itemset is in all the transactions. Consider itemset1 = {bread}
and itemset2 = {shampoo}. There will be far more transactions containing bread than those containing shampoo. So as you
rightly guessed, itemset1 will generally have a higher support than itemset2. Now consider itemset1 = {bread, butter} 
and itemset2 = {bread, shampoo}. Many transactions will have both bread and butter on the cart but bread and shampoo? 
Not so much. So in this case, itemset1 will generally have a higher support than itemset2. Mathematically,
support is the fraction of the total number of transactions in which the itemset occurs

Value of support helps us identify the rules worth considering for further analysis. For example, one might want to
consider only the itemsets which occur at least 50 times out of a total of 10,000 transactions i.e. support = 0.005.
If an itemset happens to have a very low support, we do not have enough information on the relationship between its 
items and hence no conclusions can be drawn from such a rule

**Confidence:**

This measure defines the likeliness of occurrence of consequent on the cart given that the cart already has the 
antecedents. That is to answer the question — of all the transactions containing say, {Captain Crunch}, how many also 
had {Milk} on them? We can say by common knowledge that {Captain Crunch} → {Milk} should be a high confidence rule

**Lift:**

Lift controls for the support (frequency) of consequent while calculating the conditional probability of occurrence 
of {Y} given {X}.