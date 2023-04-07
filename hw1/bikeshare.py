import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.utils import shuffle

'''
Author: Dimitrios Haralampopoulos
Pledge: I pledge my honor that I have abided by the Stevens Honor System
Overview: With the dataset given for this problem, I wanted to see how weather conditions affected the number of people who would be using bike sharing services. I chose the input variables to be
Weathersit (the type of weather on the given day/hour), Atemp (the "real feel" temperature so to speak), and Hum (the humidity outside). 
'''

day_file = pd.read_csv (r'day.csv', usecols = ["weathersit", "atemp", "hum", "casual", "registered", "cnt"])  # Retrieving same features from both sets
hour_file = pd.read_csv (r'hour.csv', usecols = ["weathersit", "atemp", "hum", "casual", "registered", "cnt"])

ddata = day_file.values # Creating Datasets for testing and training
hdata = hour_file.values

# Splitting data accordingly, a quarter of the data will be used for testing while the remaining 3/4 will be used for training. Just convention to have that kind of proportion of test/training data
dX_train, dX_test, dY_train, dY_test = train_test_split(ddata[:,0:3], ddata[:,3:6], test_size = 0.25, random_state=1)
hX_train, hX_test, hY_train, hY_test = train_test_split(hdata[:,0:3], hdata[:,3:6], test_size = 0.25, random_state=1)

# Shuffling data around pre-training
dX_shuffle, dY_shuffle = shuffle(ddata[:,0:3], ddata[:,3:6])
hX_shuffle, hY_shuffle = shuffle(hdata[:,0:3], hdata[:,3:6])
        

# We initialize our weights as random 3x3 tensors since we have 3 inputs and 3 outputs
w = np.random.random_sample((3,3))
b = np.random.random_sample((3,3))

# Creating paired training datasets for unpacking during gradient descent
d = []
for x,y in zip(dX_train, dY_train):   
    d.append((x,y))
print(d)
h = []
for x,y in zip(hX_train, hY_train):
    h.append((x,y))

# A slightly different take on the MSE function, computed as such to sum up column vectors
def dmse(yp, y, size, alpha):
    return np.sum((yp-y)**2 + (alpha*dmult()),axis=0)/size

def hmse(yp, y, size, alpha):
    return np.sum((yp-y)**2 + (alpha*hmult()),axis=0)/size

# Row-by-column dot product of the weight matrix with itself, needed for accuracy of computation of regularization term
def dmult():
    global w
    x = np.array([])
    for i in w:
        x = np.append(x, np.dot(i,i.T))
    return x
def hmult():
    global b
    x = np.array([])
    for i in b:
        x = np.append(x, np.dot(i,i.T))
    return x

# This is the resulting function of the gradient of the MSE function. 
def lsf (yp, y, x):
    return (2*np.dot(x.T, (yp-y)))

def minibatch_d (D, step, epochs, batch, alpha = 1.0):
    global w
    for i in range(epochs):
        #I have to shuffle all the data yet keep the inputs still matched with the outputs, so a single dataset solution seemed most appropriate
        trainloader = DataLoader(dataset = D, batch_size = batch, shuffle= True) 
        for x, y in trainloader:
            xt = (x.detach().cpu().numpy())     #Split x and y and convert them from tensors to numpy arrays
            yt = (y.detach().cpu().numpy())
            yhat = np.dot(xt, w)    #This gets us our estimated output given the weights and our training inputs
            lfun = lsf(yhat, yt, xt)/batch + 2*alpha*w     #This produces our loss function with regularization term added 
            w = w - (step * lfun)   #updating our weight vector here
    return w

def minibatch_h (H, step, epochs, batch, alpha = 1.0):
    global b
    for i in range(epochs):
        #I have to shuffle all the data yet keep the inputs still matched with the outputs, so a single dataset solution seemed most appropriate
        trainloader = DataLoader(dataset = H, batch_size = batch, shuffle= True) 
        for x, y in trainloader:
            xt = (x.detach().cpu().numpy())     #Split x and y and convert them from tensors to numpy arrays
            yt = (y.detach().cpu().numpy())
            yhat = np.dot(xt, b)    #This gets us our estimated output given the weights and our training inputs
            lfun = lsf(yhat, yt, xt)/batch + 2*alpha*b     #This produces our loss function with regularization term added 
            b = b - (step * lfun)   #updating our weight vector here
    return b

# Short function for repeated learning/training
def learning_d (D, step, epochs, batch, gen, alpha = 1.0):
    for i in range(gen):
        minibatch_d(D, step, epochs, batch, alpha)
        print(f"Learning dataset: Days . . . generation: {i+1}")
def learning_h (H, step, epochs, batch, gen, alpha = 1.0):
    for j in range(gen):
        minibatch_h(H, step, epochs, batch, alpha)
        print(f"Learning dataset: Hours . . . generation: {j+1}")

def predicting_d (D, T, out, alpha=1.0):    # These produce "predictions" that though are not very accurate compared to the Ridge Regression, it still understands the shape of the data
    global w                                    # or in other words, the distinct additive properties of casual, registered, and cnt are preserved
    mse = np.sum(dmse(np.dot(w.T, D.T).T, T, out, alpha)/np.shape(D)[0])/out
    #print(f'Prediction: {np.dot(w.T, D.T).T}\nActual: {T}\nMSE: {mse}')
    return mse
def predicting_h (H, T, out, alpha=1.0):
    global b
    mse = np.sum(hmse(np.dot(b.T, H.T).T, T, out, alpha)/np.shape(H)[0])/out
    #print(f'Prediction: {np.dot(b.T, H.T).T}\nActual: {T}\nMSE: {mse}')
    return mse

def fpredicting_d (D, T, out, alpha=1.0):    # These produce "predictions" that though are not very accurate compared to the Ridge Regression, it still understands the shape of the data
    global w                                    # or in other words, the distinct additive properties of casual, registered, and cnt are preserved
    mse = np.sum(dmse(np.dot(w.T, D.T).T, T, out, alpha)/np.shape(D)[0])/out
    print(f'Prediction: {np.dot(w.T, D.T).T}\nActual: {T}\nMSE: {mse}')
    return mse
def fpredicting_h (H, T, out, alpha=1.0):
    global b
    mse = np.sum(hmse(np.dot(b.T, H.T).T, T, out, alpha)/np.shape(H)[0])/out
    print(f'Prediction: {np.dot(b.T, H.T).T}\nActual: {T}\nMSE: {mse}')
    return mse

# Here we define our cross validation (Repeated K-Fold in this case). We have defined splits as 10 and repeats as 3 somewhat arbitrarily, though they are commonly used parameters for it.
# We use 30 different model fittings in this case then to estimate the model efficacy
dcv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state= 1)
hcv = RepeatedKFold(n_splits= 10, n_repeats= 3, random_state = 1)

def drkf(step, epochs, batch, alpha = 1.0):
    global w
    v = np.copy(w) # copy of w for restoration
    mse = []
    for train_x, test_x in dcv.split(dX_shuffle): # this is where we make the actual k-fold
        d = []
        dx = []
        dy = []
        tx = []
        ty = []
        for x in train_x:   # builds the components of training set
            dx.append(dX_shuffle[x])
            dy.append(dY_shuffle[x])
        for y in test_x:    # creates the testing sets
            tx.append(dX_shuffle[y])
            ty.append(dY_shuffle[y])
        for x,y in zip(dx, dy):     # builds the training set from the components
            d.append((x,y))
        minibatch_d(d, step, epochs, batch, alpha)  # mini-batch to train the model
        tx = np.array(tx)   # converting to np.array for compatibility with the math performed in the prediction
        ty = np.array(ty)
        mse.append(predicting_d(tx, ty, 3))     # yields the mse for that given prediction to the list of MSEs
    w = np.copy(v)  # resets w
    return np.mean(mse)     # mean of MSEs is computed and we return that as our cross-validation with MSE scoring
        
def hrkf(step, epochs, batch, alpha = 1.0):     # same concept as the above function except for hdata's specifications
    global b
    v = np.copy(b)
    mse = []
    for train_x, test_x in dcv.split(hX_shuffle):
        h = []
        hx = []
        hy = []
        jx = []
        jy = []
        for x in train_x:   
            hx.append(hX_shuffle[x])
            hy.append(hY_shuffle[x])
        for y in test_x:
            jx.append(hX_shuffle[y])
            jy.append(hY_shuffle[y])
        for x,y in zip(hx, hy):
            h.append((x,y))
        minibatch_h(h, step, epochs, batch, alpha)
        jx = np.array(jx)
        jy = np.array(jy)
        mse.append(predicting_h(jx, jy, 3))
    b = np.copy(v)
    return np.mean(mse)

# Learning . . .
learning_d(d, 0.1, 10, 50, 5, 1.0)
learning_h(h, 0.1, 10, 50, 5, 1.0)

# Predicting . . .
print(f"Prediction MSE for ddata: {fpredicting_d(dX_test, dY_test, 3)}")
print(f"Prediction MSE for hdata: {fpredicting_h(hX_test, hY_test, 3)}")

# Cross Validating . . .
print(f"Cross validation for ddata: {drkf(0.1, 10, 50)}")
print(f"Cross validation for hdata: {hrkf(0.1, 10, 50)}") # For some reason the cross validation for Hour takes a little while, but it should appear within 10~15 seconds
    







    

