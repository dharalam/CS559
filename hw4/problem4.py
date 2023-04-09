import numpy as np
import pandas as pd

def eucl(xi, xj):
    return np.sqrt(np.sum(np.square(xi-xj)))

data = np.array([[4.6, 2.9], [4.7, 3.2], [4.9, 3.1], [5.0, 3.0], [5.1, 3.8], [5.5, 4.2], [6.0, 3.0], [6.2, 2.8], [6.7, 3.1]])
m = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]]) # m1, m2, m3
iterations = 0

def kmeans (data, m, k):
    global iterations
    clusters = [[] for i in range(k)] # clusters to compute new means
    for i in data:
        classif = [eucl(i, m[j]) for j in range(k)] # gets classifications for each of the centers
        clusters[classif.index(min(classif))].append(i) # finds best classification and maps the point to that cluster
    means = [np.mean(clusters[i], axis = 0) for i in range(k)] # computes new cluster centers
    if False not in np.equal(m, means): # base case
        return clusters
    m = means
    iterations += 1
    print(f"<<Iter {iterations}>>: New centers Red: {np.round(m[0], 3)}, Green: {np.round(m[1], 3)}, Blue: {np.round(m[2], 3)}\n")
    return kmeans(data, m, k) # recursive step

print(kmeans(data, m, 3))