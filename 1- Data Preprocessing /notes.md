# Data Preprocessing Notes
## Notes
- Apply feature scaling **after** splitting data to prevent information leakage on test set which we're 
not supposed to have until the training is done
- Do we've to apply feature scaling to dummy variables ? __NO__
, As if we applied it to dummy vars we'll lose our interpretations to the sense of these
numerical vars 

## Feature Scaling
- *Normalisation* is recommended  when we've a normal distribution in most of our features
- *Standardisation* is working all the time
#### Standardisation
```Xstand = x - mean(x) / stdev(x)```
#### Normalisation 
```Xnorm = x - min(x) / max(x) - min(x)```

## Preprocessing Steps
1. Handling missing data
2. Encoding categorical data
3. Splitting dataset
4. Apply feature scaling if required to needed features

