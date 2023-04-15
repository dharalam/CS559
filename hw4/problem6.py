import tensorflow as tf
import pickle, gzip
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding="latin 1")
f.close()

ips = (28, 28, 1) # will use later

train_data = train_set[0]
train_class = train_set[1]

valid_data = valid_set[0]
valid_class = valid_set[1]

test_data = test_set[0]
test_class = test_set[1]

train_data = train_data.reshape(50000, 28, 28, 1) # reshaping back to an nx28x28x1 tensor for ease of use with packages
test_data = test_data.reshape(10000, 28, 28, 1)
valid_data = valid_data.reshape(10000, 28, 28, 1)

train_class = tf.one_hot(train_class.astype(np.int32), depth = 10) # one-hot to include categorical datset in the model
test_class = tf.one_hot(test_class.astype(np.int32), depth = 10)
valid_class = tf.one_hot(valid_class.astype(np.int32), depth = 10)

batch_size = 50 # change whenever
num_classes = 10 # should remain 10 classes ([0,1,2,3,4,5,6,7,8,9] = 10 classes)
epochs = 5 # change whenever

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (5,5), padding="same", activation="relu", input_shape = ips),  # 32 filter convolution, applied to 5x5 portions of the image
                                    tf.keras.layers.Conv2D(32, (5,5), padding="same", activation="relu"), # ^^^
                                    tf.keras.layers.MaxPool2D(), # reduces a full-size matrix of the image to a single pixel with a maximum value
                                    tf.keras.layers.Dropout(0.25), # randomly ignore 25% of the nodes in the layer, helps learn different features
                                    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),  # 64 filter convolution, applied to 3x3 portions of the image
                                    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"), # ^^^
                                    tf.keras.layers.MaxPool2D(strides = (2,2)), # reduces a 2x2 matrix of the image to a single pixel with a maximum value
                                    tf.keras.layers.Dropout(0.25), # ignore 25% of nodes at random
                                    tf.keras.layers.Flatten(), # flattens tensors to 1D vector
                                    tf.keras.layers.Dense(128, activation="relu"), # Creates an 128-node artificial neural network with relu activation
                                    tf.keras.layers.Dropout(0.5), # ignore 50% of nodes at random
                                    tf.keras.layers.Dense(num_classes, activation="softmax")]) # 10-node ANN (with softmax to top it off), returns the probability that an image is in each class

model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss="categorical_crossentropy", metrics=["acc"]) # categorical since we want to categorize/classify results

class CallBack(tf.keras.callbacks.Callback):  # Early Stopping
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.995):
            print("\nReached 99.5% accuracy, applying early stopping.\n")
            self.model.stop_training = True

callbacks = CallBack()

history = model.fit(train_data, train_class, batch_size=batch_size, epochs=epochs, validation_data=(valid_data, valid_class), 
                    validation_batch_size = batch_size, callbacks=[callbacks]) # fits model and runs validation data alongside it

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history["loss"], color='b', label = "Training Loss")
ax[0].plot(history.history["val_loss"], color='r', label = "Validation Loss", axes = ax[0])
legend = ax[0].legend(loc="best", shadow = True)
ax[1].plot(history.history["acc"], color='b', label = "Training Accuracy")
ax[1].plot(history.history["val_acc"], color='r', label = "Validation Accuracy")
legend = ax[1].legend(loc="best", shadow = True)
plt.show()
tloss, tacc = model.evaluate(test_data, test_class)






