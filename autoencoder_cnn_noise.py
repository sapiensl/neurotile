import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = xTrain.astype("float32") / 255.
xTest = xTest.astype("float32") / 255.

noiseFactor = 0.2
xTrain_noise = xTrain + noiseFactor * tf.random.normal(shape=xTrain.shape)
xTest_noise = xTest + noiseFactor * tf.random.normal(shape=xTest.shape)

xTrain_noise = tf.clip_by_value(xTrain_noise, clip_value_min=0., clip_value_max=1.)
xTest_noise = tf.clip_by_value(xTest_noise, clip_value_min=0., clip_value_max=1.)

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28,28,1)),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same", strides=2),
            layers.Conv2D(8, (3, 3), activation="relu", padding="same", strides=2)])
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation="relu", padding="same"),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation="relu", padding="same"),
            layers.Conv2D(1,kernel_size=(3,3), activation="sigmoid", padding="same")])
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
denoiseModel = Denoise()

denoiseModel.compile(optimizer="adam", loss=losses.MeanSquaredError())

denoiseModel.fit(xTrain_noise[:10000], xTrain[:10000], epochs=5, shuffle=True, validation_data=(xTest_noise, xTest))

plt.figure(figsize=(20,6))
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(xTest[i])
    plt.gray()
    plt.subplot(3,10,i+1+10)
    plt.imshow(xTest_noise[i])
    plt.gray()
    
    processedImage = denoiseModel(xTest_noise[i:i+1])[0]
    plt.subplot(3,10,i+1+20)
    plt.imshow(processedImage)
    plt.gray()
plt.show()
    