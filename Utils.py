import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import Parser
import KMeans


"""
Apply Principal Component Analysis to instances and centroids.
Returns the data and the centroids in the two dimensions with highest variance as numpy arrays.
Input:
	-instances is a numpy ndarray with the instances
	-centroids is a numpy array of centroids
"""
def pca(instances, centroids):
	# Transform instances
	instances = instances.T

	mean = np.mean(instances, axis=1)
	mean = mean.reshape(26, 1)
	reescaled_instances = instances - mean

	cov_matrix = np.cov(reescaled_instances)
	eig_val, eig_vect = np.linalg.eig(cov_matrix)

	eigen_pair = [(np.abs(eig_val[i]), eig_vect[:, i]) for i in range(len(eig_val))]
	eigen_pair.sort()
	eigen_pair.reverse()

	matrix_w = np.hstack((eigen_pair[0][1].reshape(26, 1), eigen_pair[1][1].reshape(26, 1)))

	transformed_instances = np.dot(reescaled_instances.T, matrix_w).T

	# Transform centroids
	centroids = centroids.T

	mean = np.mean(centroids, axis=1)
	mean = mean.reshape(26, 1)
	reescaled_centroids = centroids - mean

	transformed_centroids = np.dot(reescaled_centroids.T, matrix_w).T

	return transformed_instances, transformed_centroids


"""
Plot the data transformed by PCA. Plot it in several plots, split by centroid and also plot all the data together.
Input:
	-transformed is the data transformed by PCA (numpy.narray)
	-labels is the centroid assignment of each instance
	-centroids is a numpy array of centroids
"""
def plot_transformed(transformed, labels, centroids, plots_folder):
	xMax = np.max(transformed[0, :])
	xMin = np.min(transformed[0, :])
	yMax = np.max(transformed[1, :])
	yMin = np.min(transformed[1, :])

	labels_number = list(set(labels))
	labels_number.sort()
	labels = np.array(labels).reshape(1, len(labels))

	transformed = np.vstack([transformed, labels])

	black = '#000000'
	colors = ['#FFB6C1', '#FF0000', '#FFA500', '#FFFF00', '#008000', '#00FFFF', '#0000FF', '#FF00FF', '#800080',
			  '#DAA520', '#20B2AA']

	# Plot each cluster separated
	for i in labels_number:
		filter = transformed[2, :] == i
		transformed_filtered = transformed[0:2, filter]
		plt.figure()
		plt.plot(transformed_filtered[0, :], transformed_filtered[1, :], 'o', markersize=7, color=colors[i], alpha=0.5)
		plt.plot(centroids[0, i], centroids[1, i], 'x', markersize=8, color=black, alpha=0.5)
		plt.title('Cluster del centroide {centroid}'.format(centroid=centroids[:, i]))
		plt.xlim([xMin, xMax])
		plt.ylim([yMin, yMax])
		plt.savefig(plots_folder+"/"+('cluster{num}.png'.format(num=i)))

	# Plot all clusters together
	plt.figure()
	for i in labels_number:
		filter = transformed[2, :] == i
		transformed_filtered = transformed[0:2, filter]
		plt.plot(transformed_filtered[0, :], transformed_filtered[1, :], 'o', markersize=7, color=colors[i], alpha=0.5)
	for i in labels_number:
		plt.plot(centroids[0, i], centroids[1, i], 'x', markersize=8, color=black, alpha=0.5)
	plt.title('Clusters')
	plt.xlim([xMin, xMax])
	plt.ylim([yMin, yMax])
	plt.savefig(plots_folder+"/"+'clusters.png')

if __name__ == '__main__':
	instances = Parser.parse_data()
	assignment = KMeans.kmeans(instances, 10, 0.01)
	centroids = np.array(list(assignment.keys()))

	labels = KMeans.get_centroid_labels(instances, assignment)

	transformed_instances, transformed_centroids = pca(instances, centroids)

	plot_transformed(transformed_instances, labels, transformed_centroids)
