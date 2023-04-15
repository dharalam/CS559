import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")
data = open(r"multigauss.txt", "r")

pX = (data.read()).split("\n")
pX = [pX[i].split(" ")[1::] for i in range(len(pX))]
X = [[float(pX[i][0]), float(pX[i][1])] for i in range(len(pX)-1)]
X = np.asmatrix(X)

def gauss (X, k, mu, cov):
    return np.exp(np.dot(np.dot(-0.5*(X-mu), np.linalg.inv(cov)), (X-mu).T))/np.sqrt(((2*math.pi)**k)*np.abs(np.linalg.det(cov)))
    

def Expectation(X, k, pi, mu, cov):
    w = [[(pi[j] * gauss(i, k, mu[j], cov[j]))/np.sum([pi[h] * gauss(i, k, mu[h], cov[h]) for h in range(k)]) for j in range(k)] for i in X]
    return np.asarray(w).reshape(X.shape[0], k)

def MaximizeMean(X, k, w):
    mu = []
    for i in range(k):
        N = np.sum([np.multiply(w[j][i], X[j]) for j in range(X.shape[0])], axis = 0)
        Nk = np.sum([w[j][i] for j in range(X.shape[0])], axis = 0)
        mu.append(N/Nk)
    return np.asarray(mu).reshape(k, X.shape[1])        

def MaximizeCovariance(X, k, w, mu):
    cov = []
    for i in range(k):
        N = [np.multiply(w[j][i], np.dot((X[j]-mu[i]).T, (X[j]-mu[i]))) for j in range(X.shape[0])]
        N = np.sum(N, axis = 0)
        Nk = np.sum([w[j][i] for j in range(X.shape[0])], axis = 0)
        cov.append(N/Nk)
    return np.asarray(cov)
    

def MaximizeMixtures(k, w):
    pi = []
    for i in range(k):
       Nk = np.sum([w[j][i] for j in range(X.shape[0])], axis = 0)
       N = w.shape[0]
       pi.append(Nk/N)
    return np.asarray(pi).reshape(k, 1)

def lhood(X, k, nPi, nMu, nCov):
    logsum = []
    for i in range(X.shape[0]):
        N = np.sum([nPi[h] * gauss(i, k, nMu[h], nCov[h]) for h in range(k)])
        logsum.append(np.log(N))
    return np.sum(np.asarray(logsum))

def EM(X, k, nIter):
    results = []
    pi0 = np.random.rand(k, 1)
    mu0 = np.random.rand(k, X.shape[1])
    cov0 = np.asarray([np.random.rand(X.shape[1], X.shape[1]) for i in range(k)])
    print(pi0.shape)
    print(mu0.shape)
    print(cov0.shape)
    for i in range(nIter):
        w = Expectation(X, k, pi0, mu0, cov0)
        print(w.shape)
        nMu = MaximizeMean(X, k, w)
        print(nMu.shape)
        nCov = MaximizeCovariance(X, k, w, nMu)
        print(nCov.shape)
        nPi = MaximizeMixtures(k, w)
        print(nPi.shape)
        lh = lhood(X, k, nPi, nMu, nCov)
        results.append(lh)
        if len(results) > 1 and (results[-1] == results[-2]):
            return nPi, nMu, nCov
        pi0 = nPi
        mu0 = nMu
        cov0 = nCov
    return nPi, nMu, nCov

rPi, rMu, rCov = EM(X, 3, 5)
print(f"Final Pi:\n{rPi}\nFinal Mu:\n{rMu}\nFinal Cov:\n{rCov}") 