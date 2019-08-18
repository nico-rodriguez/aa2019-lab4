import KMeans
import Parser
from sklearn.metrics import silhouette_score, adjusted_rand_score

"""
Evaluates the KMeans algorithm using Silhouette Score.
Input:
	- data is the array of instances (only the answer section).
	- k is the k coefficient in the KMeans algorithm.
"""
def evaluate_silhouette_score(data, k, threshold, random):
	assignment = KMeans.kmeans(data, k, threshold, random)
	labels = KMeans.get_centroid_labels(data, assignment)
	return silhouette_score(data, labels, metric='euclidean')


"""
Evaluates the KMeans algorithm using Adjusted Rand Index. The evaluation
is against the party label of the instances. Since there are 11 parties,
the k coefficient of KMeans algorithm is 11.
Input:
	- labels_true is a list with the party values of each instance.
"""
def evaluate_adjusted_rand_score(data, threshold, random):
	full_data = Parser.parse_data(only_answers=False)
	labels_true = Parser.get_true_party_assignment(full_data)
	number_of_parties = 11
	assignment = KMeans.kmeans(data, number_of_parties, threshold, random)
	labels_pred = KMeans.get_centroid_labels(data, assignment)
	return adjusted_rand_score(labels_true, labels_pred)


if __name__ == '__main__':
	data = Parser.parse_data()
	number_of_instances = len(data)
	print(evaluate_silhouette_score(data, 10, 0.01, True))
	print(evaluate_adjusted_rand_score(data, 0.01, True))