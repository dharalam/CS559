import numpy as np
import pandas as pd
import math

data = open(r"multigauss.txt", "r")

pX = (data.read()).split("\n")
pX = [pX[i].split(" ")[1::] for i in range(len(pX))]
X = [[float(pX[i][0]), float(pX[i][1])] for i in range(len(pX)-1)]
X = np.asmatrix(X)

def gauss (X, k, mu, cov):
    return np.exp(np.dot(np.dot(-0.5*(X-mu).T, np.linalg.inv(cov)), (X-mu)))/np.sqrt(((2*math.pi)**k)*np.linalg.det(cov))
    

def Expectation(X, k, pi, mu, cov):
    w = [[(pi[j] * gauss(i, k, mu[j], cov[j]))/np.sum([pi[h] * gauss(i, k, mu[h], cov[h]) for h in range(k)]) for j in range(k)] for i in X]
    return w

def MaximizeMean(X, k, w):
    Nk = np.sum(w)
    return np.sum(np.multiply(w, X))/Nk

def MaximizeCovariance(X, k, w, mu):
    Nk = np.sum(w)
    return np.sum(np.dot(np.dot(w, (X-mu)), (X-mu).T))/Nk

def MaximizeMixtures(k, w):
    return 0

def EM(X, k, pi0, mu0, cov0, nIter):
    return 0