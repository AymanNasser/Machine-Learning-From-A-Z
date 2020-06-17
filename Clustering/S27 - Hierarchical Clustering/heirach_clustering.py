from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import scipy.cluster.hierarchy as sch
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Loading data set
data_frame = pd.read_csv('../Mall_Customers.csv')
data_frame = data_frame.drop('CustomerID', axis=1)

X = data_frame.iloc[:, [2,3]].values # Features

# Creating a Dendrogram
# Methods: str, default is 'single'. Types of methods: 'single', 'average', 'weighted', 'centroid', 'median' & 'ward'
# method = ’ward’ uses the Ward variance minimization algorithm
dendrogram = sch.dendrogram( sch.linkage(X, method = 'ward') )
plt.title('Dendrogram')
plt.xlabel('Customers ID')
plt.ylabel('Distances between each pair')
plt.show()

# Creating Hierarchical cluster with optimal no.of clusters
# We chose no_clusters = 5 from dendrogram insights
# affinity: Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
# If linkage is “ward”, only “euclidean” is accepted
no_clusters = 5
hier_cluster = AgglomerativeClustering(n_clusters=no_clusters, linkage= 'ward',affinity= 'euclidean')
# fit_predict(): Compute cluster centers and predict cluster index for each sample.
y_hier = hier_cluster.fit_predict(X)

# Visualizing clusters
# Selecting X-Axis as annual income & Y-axis as spending score
# S= size of points
for clusters in range(0,no_clusters):

    r = random.random()
    b = random.random()
    g = random.random()
    color = (r,g,b)
    plt.scatter(X[y_hier == clusters, 0], X[y_hier == clusters, 1], s=50, c=color, label = 'Cluster {}'.format(clusters+1))

plt.title('Customer Segmentation')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()

