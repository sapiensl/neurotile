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

plt.figure(figsize=(10,10))
for i in range(10):
    fig = plt.subplot(10,10, i+1)
    plt.imshow(xTest[i])
    plt.gray()

for hiddenN in range(1,10):
    autoencoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(hiddenN, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
        layers.Reshape((28,28))
        ])

    autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

    autoencoder.fit(xTrain,xTrain, epochs=5, shuffle=True, validation_data=(xTest,xTest))

    for i in range(10):
        processedImage = autoencoder(xTest[i:i+1])
        fig = plt.subplot(10,10, i+1+hiddenN*10)
        plt.imshow(processedImage[0])
        plt.gray()
plt.show()
