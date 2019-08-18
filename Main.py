"""
Main module of the project.
"""

import Evaluator
import KMeans
import numpy as np
import Parser
import sys
import Utils
import os

uso_general = """
Invocar como:

python3 Main.py [k] [Threshold] [Random(T o F)] [Plot(T o F)]

donde:
- k indica la cantidad de centroides a utilizar en la ejecución.
- Threshold indica el umbral de error máximo de convergencia al correr K-Means.
- Random: indicar ingresando "T" para que la elección de los centroids sea aleatoria, "F" para que no.
- Plot: indica si se genera o no plots para la evaluación.
"""

plots_folder = "clusters_plots"

if __name__ == "__main__":
    # checking arguments
    if (len(sys.argv) < 5):
        print('#########################')        
        print('Error. Cantidad de parametros invalidos')
        print('-------------------------')
        print(uso_general)
        exit()
    k = sys.argv[1]
    threshold = sys.argv[2]
    random = sys.argv[3]
    plot = sys.argv[4]
    #Check if k is valid
    if not k.isdigit():
        print('#########################')        
        print('Error. El k ingresado no es válido')
        print('-------------------------')
        exit()
    else:
        k = int(k)
    #Check if threshold is valid
    try:
        threshold = float(threshold)
    except ValueError:
        print('#########################')        
        print('Error. Threshold inválido, ingrese un entero o decimal.')
        print('-------------------------')
        exit()
    #Check if random is valid
    if random != "T" and random != "F":
        print('#########################')
        print('Error. Modo de uso inválido.')
        print('-------------------------')
        print(uso_general)
        exit()
    else:
        if random is "T":
            random = True
        else:
            random = False
    #Check if plot is valid
    if plot != "T" and plot != "F":
        print('#########################')
        print('Error. Modo de uso inválido.')
        print('-------------------------')
        print(uso_general)
        exit()
    else:
        if plot is "T":
            plot = True
        else:
            plot = False
    if k < 0:
        print('#########################')
        print('Error. Valor de k negativo, o cero. Debe ser positivo')
        print('-------------------------')
        exit()
    print('Parseando conjunto de datos')
    instances = Parser.parse_data()
    print('Se completó el parseo de los datos')
    print('Comenzando ejecución del algoritmo K-Means para el conjunto de datos "aquienvoto.uy" (puede tardar unos minutos)')
    assignment = KMeans.kmeans(instances, k, threshold, random)
    print('Fin de ejecución del algoritmo K-Means')
    if plot:
        print('Comenzando plots')
        centroids = np.array(list(assignment.keys()))
        labels = KMeans.get_centroid_labels(instances, assignment)
        transformed_instances, transformed_centroids = Utils.pca(instances, centroids)
        if not os.path.isdir(plots_folder):
            os.mkdir(plots_folder)
        Utils.plot_transformed(transformed_instances, labels, transformed_centroids, plots_folder)
        print('Se generaron los plots de los clusters en la carpeta actual satisfactoriamente')
    print('Evaluando coeficiente de Silhouette con resultado de K-means(puede tardar unos minutos):')
    print('Silhouette Score:'+str(Evaluator.evaluate_silhouette_score(instances, k, threshold, random)))
    print('Evaluando Adjusted Rand Score con resultado de K-means(puede tardar unos minutos):')
    print('Adjusted Rand Score:'+str(Evaluator.evaluate_adjusted_rand_score(instances, threshold, random)))
    print('El Main termino de ejecutar de forma exitosa.')