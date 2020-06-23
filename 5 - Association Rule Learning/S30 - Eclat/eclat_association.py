import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

# header=None ==> Specifying the function that the dataset has no headers to contain the first row
dataset = pd.read_csv('../Market_Basket_Optimisation.csv', header=None)

list_transactions = []
# Inserting each transaction to a list to train our model on
for rows in range(0,len(dataset)):
    list_transactions.append( [ str(dataset.values[rows,cols]) for cols in range(0,20) ] )

# min_support = 3 frequent items per day * 7 days as the dataset is taken each week / number of transactions per week
# play with min_confidenece
rules = apriori(transactions = list_transactions, min_support= 3*7 / len(dataset), min_confidence= 0.2, min_lift= 3,
                min_length = 2, max_length= 2)

results = list(rules)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending supports
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))

