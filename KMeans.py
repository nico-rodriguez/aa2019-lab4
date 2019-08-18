import numpy as np
import Parser
import time
import random

"""
Applies kmeans on the dataset data as input
Input:
	data is a panda dataset
	k is the ammount of centroids
	threshold is the error tolerance in the distance between each
		centroid and its past version.
Output:
	a dict with centroids as keys and the elements of the dataset that
	belong to each of those centroids as values
	(represented as a numpy array)
"""
def kmeans(data, k, threshold, pure_random=True):
	centroids = obtain_random_centroids(data, k, pure_random)
	past_centroids = None
	while not convergence(centroids, past_centroids, threshold):
		assignment = assign_datapoints_to_centroids(centroids, data)
		past_centroids = centroids
		centroids = update_centroids(assignment, centroids)
	return assignment


"""
Return the labels of each instance as a list of labels.The label is just a number (starting at 0)
that identifies a centroid. The ith entry of the list is the label of the ith instance.
Input:
	- data is an array with the instances.
	- assignment is a dictionary with centroids as keys and instances as values (arrays)
"""
def get_centroid_labels(data, assignment):
	labels = []
	centroids = list(assignment.keys())
	for i in range(len(data)):
		# Label the ith instance
		for j in range(len(centroids)):
			if any((assignment[centroids[j]][:] == data[i]).all(1)):
				# Label the ith instance with the jth centroid.
				labels.append(j)
	return labels


"""
Returns a numpy array of points, where the minimum
and maximum range of values for each element of the centroids is
obtained from data.
The data space is divided in kquantiles. Each centroid is in its respective
quantile. The amount of quantiles considered is k+1, so that we dont take
the 100% and 0% quantiles into account
Input:
	data is a numpy array with dataset
	k is the ammount of centroids
Output:
	list of centroids, each centroid is represented by a tuple
"""
def obtain_random_centroids(data,k, pure_random):
	result_aux = []
	# we first need to redimension data
	data_transposed = data.T
	for i in range (0, k): # for each k centroid
		# we create that centroid with some points for each of its 26 dimensions
		centroid = [] 
		for j in range(0,len(data_transposed)):
			if pure_random:
				#obtain random points for each dimension
				point = float(random.randint(0, 5))
				centroid.append(point)
			else:
				# the space is divided in k+1 partitions, 
				# data points leave exactly 1/k+1 points to their sides
				point = float(np.quantile(data_transposed[j], (i+1)/(k+1)))
				centroid.append(point)
		result_aux.append(centroid)
	result = np.array(result_aux)
	return result


"""
Returns true if the difference in norm of the past and actual centroids
is lower than threshold for each centroid 
Input:
	centroids are the actual centroids
	past_centroids are the centroids of the last iteration
	threshold is the error tolerance in the distance between each
	centroid and its past version.
"""
def convergence(centroids, past_centroids, threshold):
	# base case
	if past_centroids is None:
		return False
	# for each centroid
	for i in range(0, len(centroids)):
		centroid = np.array(centroids[i])
		past_centroid = np.array(past_centroids[i])
		# we calculate distance
		distance = np.linalg.norm(centroid - past_centroid)
		if distance > threshold:  # if at least one distance is greater than threshold, we return False
			return False
	return True


"""
Returns a dict in which each key is a centroid and each value
is the elements of the dataset that belong to that centroid
Input:
	centroids are the actual centroids
	data is a numpy array with the answer values of the data
"""
def assign_datapoints_to_centroids(centroids, data):
	centroid_assigment = {}
	# Initialize the dictionary
	for centr in centroids:
		centroid_assigment[tuple(centr)] = []

	for instance in data:
		# Find closest centroid
		centroids_dist = centroids - instance
		# Efficient dot product
		centroids_dist = np.einsum('ij,ij->i', centroids_dist, centroids_dist)
		closest_centroid = centroids[np.argmin(centroids_dist)]
		centroid_assigment[tuple(closest_centroid)].append(instance)

	# It's more efficient to accumulate the instances in a list first and then convert the lists to numpy arrays
	for centr in centroids:
		centroid_assigment[tuple(centr)] = np.array(centroid_assigment[tuple(centr)])

	return centroid_assigment


"""
Returns a new numpy array of centroids based on the new assignment by 
averaging the value of each element of the centroid (attributes)
Input:
	assignment is a dict in which each key is a centroid and each value
	is the elements of the dataset that belong to that centroid
"""
def update_centroids(assignment, centroids_order):
	new_centroids = np.array([])
	#Go over the centroids and generate the new one
	for centroid in centroids_order:
		#Obtain the assigned data to that centroid
		tuples_for_centroid = assignment[tuple(centroid)]
		#Get a new array that contains on the index i
		#The mean of the attribute i in all the data for that centroid
		new_centroid = tuples_for_centroid.mean(axis=0)
		#If there is still no centroids added set the obtained one
		if new_centroids.size == 0:
			new_centroids = new_centroid
		#Else add the obtained centroid to the array
		else:
			new_centroids = np.vstack((new_centroids,new_centroid))
	return new_centroids


# For debugging
def instances_number(centroid_assigment):
	instances_number = 0
	for centr in centroid_assigment:
		instances_number += len(centroid_assigment[centr])
	return instances_number


if __name__ == "__main__":
	data = Parser.parse_data()
	number_of_instances = len(data)
	assert number_of_instances == 32447

	# Test KMeans algorithm
	elapsed_time = time.time()
	assignment = kmeans(data, 10, 0.1)
	elapsed_time = time.time() - elapsed_time
	print(assignment)
	print('instances number: ', instances_number(assignment))
	assert instances_number(assignment) == number_of_instances
	print('time elapsed: ', elapsed_time, 's')

	# Test get_centroid_labels function
	labels = get_centroid_labels(data, assignment)
	print(labels)
	assert len(labels) == number_of_instances
