# Laboratorio 4 del curso Aprendizaje Automático 2019.

## Dependencias
* numpy >= 1.16.3 : https://github.com/numpy/numpy
* pandas >= 0.24.2 : https://github.com/pandas-dev/pandas
* python3 >= 3.5.2
* scikit-learn >= 0.22 : https://github.com/scikit-learn/scikit-learn

Para instalar la última versión de `scikit-learn`,`pandas`,`numpy` como dependencia de python3 ejecutar `pip3 install [--user] library_name`.

También, se utilizan paquetes que por defecto vienen con Python: numbers, random, sys, time.

### Análisis de Componentes Principales
El cóódigo fuente para el PCA se encuentra en un Notebook de Python en el directorio ```data/```.

### Evaluar K-Means
Para Evaluar el algoritmo de *K-Means*, con las métricas Silhouette y Adjusted Rand Index invocar como:

python3 Main.py [k] [Threshold] [Random] [Plot]

donde:
- k indica la cantidad de centroides a utilizar en la ejecución (int).
- Threshold indica el umbral de error máximo de convergencia al correr K-Means (int ó float).
- Random: indicar ingresando "T" para que la elección de los centroids sea aleatoria, "F" para que no. ("T" o "F")
- Plot: indica si se genera o no plots para la evaluación, con "T" genera los plots y con "F" no. ("T" o "F")

Por ejemplo la invocación: python3 Main.py 11 0.1 T T

Invoca Main.py con k = 11, threshold = 0,1, centroides iniciales generados de forma aleatoria y generando plots de los resultados de K-Means.

## Archivos generados
Para el caso en el que se invoque con plot se generara un archivo para cada cluster como se describe a continuación.

```
+-- clusters_plots
|   +-- cluster1.png
|   +-- cluster2.png
|   +-- ...
|   +-- clusters.png
```

