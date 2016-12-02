import numpy as np
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

bandwidth = estimate_bandwidth(X,quantile=0.18)

ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)
#Array of the Cluster Centers created in the K-Means Fit method
cluster_centers = ms.cluster_centers_
labels 		= 	ms.labels_
#Extract the unique cluseter labels
n_clusters_ = len(np.unique(labels))
colors = cycle('grcbk')


plt.figure(1)
plt.clf()
#Iterate over the clusters, colors and featuresets and plot results
for k, col in zip(range(n_clusters_), colors):
	my_members = labels == k
	cluster_center = cluster_centers[k]
	plt.plot(X[my_members, 0], X[my_members, 1], col + '.', markersize=25, zorder=1)
	plt.scatter(cluster_center[0], cluster_center[1], marker='x', s=150, linewidths=5, zorder=2)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.ylabel('Sepal Width')
plt.xlabel('Sepal Length')
plt.show()