import numpy as np
import pandas as pd

def eucl(xi, xj):
    return np.sqrt(np.sum(np.square(xi-xj)))

data = [[4.6, 2.9], [4.7, 3.2], [4.9, 3.1], [5.0, 3.0], [5.1, 3.8], [5.5, 4.2], [6.0, 3.0], [6.2, 2.8], [6.7, 3.1]]
m = [[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]] # m1, m2, m3

def kmeans (data, m):
    
