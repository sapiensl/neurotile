import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist


(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = xTrain.astype("float32") / 255.
xTest = xTest.astype("float32") / 255.

# plt.imshow(xTrain[0])
# plt.show()

autoencoder = tf.keras.Sequential([
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(784, activation="sigmoid"),
    layers.Reshape((28,28))
    ])

autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

autoencoder.fit(xTrain,xTrain, epochs=10, shuffle=True, validation_data=(xTest,xTest))

plt.figure(figsize=(20,2))
for i in range(10):
    fig = plt.subplot(2,10, i+1)
    plt.imshow(xTest[i])
    plt.gray()
    
    processedImage = autoencoder(xTest[i:i+1])
    fig = plt.subplot(2,10, i+1+10)
    plt.imshow(processedImage[0])
    plt.gray()
plt.show()