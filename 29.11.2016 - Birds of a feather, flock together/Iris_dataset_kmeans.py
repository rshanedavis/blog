import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = KMeans(n_clusters=3)
clf.fit(X)
#Array of the Centroids created in the K-Means Fit method
centroids = 	clf.cluster_centers_
#Array of the Labels created in the K-Means Fit method
labels 		= 	clf.labels_
#Array of colors for the featuresets
colors = ["g.","r.","c.","b.","k.","o."]
#Iterate over the predictions and compare to our Y label
correct = 0.0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float))
	predict_me = predict_me.reshape(-1,len(predict_me))
	prediction = clf.predict(predict_me)
	#Compare the prediction to our known labels as Y
	if prediction[0] == y[i]:
		correct += 1
	#Plot our featuresets with lables dtermined by K-means
	plt.plot( X[i][0], X[i][1] ,colors[labels[i]] , markersize=25,zorder=1)
#Plot the calculated centroids
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5 ,zorder=2)
#Print our accuracy score (remember to invert this if lower than 50%)
print(correct/len(X))

plt.ylabel('Sepal Width')
plt.xlabel('Sepal Length')
plt.show()