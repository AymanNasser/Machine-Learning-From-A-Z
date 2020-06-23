from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Loading data set
data_frame = pd.read_csv('../Mall_Customers.csv')
data_frame = data_frame.drop('CustomerID', axis=1)

# Working only with 2 features for visualization purposes
#print(data_frame.head())
#X = data_frame.iloc[:, :].values # Features
X = data_frame.iloc[:, [2,3]].values # Features


# # Encoding categorical data
# col_transform = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')
# X = np.array(col_transform.fit_transform(X))

# Finding the optimal number of clusters
wcss = []

# Trying 10 clusters
for clusters in range(1,11):

    kmean_cluster = KMeans(n_clusters= clusters, init= 'k-means++', random_state=0, max_iter= 300)
    kmean_cluster.fit(X=X)
    # inertia_: Sum of squared distances of samples to their closest cluster center.
    wcss.append(kmean_cluster.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Elpow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Creating K-means cluster with optimal no.of clusters
kmean_cluster = KMeans(n_clusters=5, init='k-means++', random_state=0, max_iter=300)
# fit_predict(): Compute cluster centers and predict cluster index for each sample.
y_kmeans = kmean_cluster.fit_predict(X)

# Visualizing clusters
# Selecting X-Axis as annual income & Y-axis as spending score
# S= size of points
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='black', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='yellow', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c='grey', label = 'Cluster 5')
plt.title('Customer Segmentation')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.scatter(kmean_cluster.cluster_centers_[:, 0], kmean_cluster.cluster_centers_[:, 1], c='green', s=100, label = 'Cluster Centroids')
plt.legend()
plt.show()

