import numpy as np
import pandas as pd
from scipy.io import arff
yeasttrain = arff.loadarff("yeast_train.arff")
train = pd.DataFrame(yeasttrain[0])
trainclass = train.pop("class")
yeasttest = arff.loadarff("yeast_test.arff")
test = pd.DataFrame(yeasttest[0])
testclass = test.pop("class")


def eucl(xi, xj):
    return np.sqrt(np.sum(np.square(xi-xj)))

def freq(List):
    return max(set(List), key = List.count)

def knn (x, train, trainclass, K):
    dist = []
    tclass = []
    for i in range(len(train.index)):
        dist.append(eucl(train.iloc[i], x))
    for j in range(K):
        cur = min(dist)
        tclass.append(trainclass.iloc[dist.index(cur)])
        dist.remove(cur)
    return freq(tclass)

def crossval (train, trainclass, test, testclass, K):
    results = []
    corr = 0
    wrong = 0
    for i in range(len(test.index)):
        results.append(knn(test.iloc[i], train, trainclass, K))
        if results[i] == testclass.iloc[i]:
            corr += 1
        else:
            wrong += 1
    accuracy = corr / (corr + wrong)
    print("For {:d}-Nearest-Neighbors, accuracy of {:0.2f}% was achieved.\n".format(K, (accuracy*100)))
    return accuracy

def modelselect(train, trainclass, test, testclass, types):
    print("Beginning model evaluation...")
    accuracies = [crossval(train, trainclass, test, testclass, i) for i in types]
    maxac = max(accuracies)
    model = types[accuracies.index(maxac)]
    print("Best model is {:d}-Nearest-Neighbors with accuracy {:0.2f}%.\n".format(model, maxac*100))
    return model

modelselect(train, trainclass, test, testclass, [1, 3, 5, 10, 15, 20, 50])
