import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision

import os
import shutil
import random
import PIL
import PIL.Image
import threading
import csv
import traceback
import json

upscaleModel = None
upscaleDiscModel = None
optimizer = optimizers.Adam(0.0001)
discOptimizer = optimizers.Adam(0.0001)
    
class TextureUpscaler(Model):
    def __init__(self, numChannels):
        super(TextureUpscaler, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (7,7), strides=1, padding="valid", input_shape=(None, None, numChannels)),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (5,5), strides=1, padding="valid"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (5,5), strides=1, padding="valid"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(numChannels, (3,3), strides=1, padding="same", activation="sigmoid")
            ])
        
    def call(self, x, training=False):
        result = tf.pad(x, tf.constant([[0,0],[7, 7], [7, 7],[0,0]]), "REFLECT")
        result = self.model(result)
        return result

class TextureUpscalerDiscriminator(Model):
    def __init__(self, numChannels):
        super(TextureUpscalerDiscriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (4,4), strides=2, padding="same", input_shape=(None, None, numChannels)),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (4,4), strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, (4,4), strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, (4,4), strides=1, padding="same", activation="sigmoid")
            ])
        
    def call(self, x, training=False):
        result = self.model(x)
        return result


def trainStep(originalImages):
    global upscaleModel
    #print("trainStep(...) was retraced!")
    
    with tf.GradientTape() as tape, tf.GradientTape() as discTape:
        smallImages = tf.image.resize(originalImages, (
                                                            originalImages.shape[1]//2,
                                                            originalImages.shape[2]//2))
        
        upscaledImages = upscaleModel(smallImages, training=True)
        
        realDiscOutput = upscaleDiscModel(originalImages)
        fakeDiscOutput = upscaleDiscModel(upscaledImages)
        
        ladvLoss = losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(fakeDiscOutput), fakeDiscOutput)
        loss = 0.001*ladvLoss + losses.mean_squared_error(originalImages, upscaledImages)
        #loss = losses.mean_squared_error(originalImages, upscaledImages)
        
        discLoss = losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(fakeDiscOutput), fakeDiscOutput) + losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(realDiscOutput), realDiscOutput) 
        
        gradients = tape.gradient(loss, upscaleModel.trainable_variables)
        discGradients = discTape.gradient(discLoss, upscaleDiscModel.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, upscaleModel.trainable_variables))
        discOptimizer.apply_gradients(zip(discGradients, upscaleDiscModel.trainable_variables))
        
def createUpscaleModel(numChannels=3):
    global upscaleModel
    global upscaleDiscModel
    upscaleModel = TextureUpscaler(numChannels)
    upscaleDiscModel = TextureUpscalerDiscriminator(numChannels)
    
def saveWeights(path):
    global upscaleModel
    global upscaleDiscModel
    
    upscaleModel.model.save_weights(path+"up/generator")
    upscaleDiscModel.model.save_weights(path+"up/discriminator")
    
def loadWeights(path):
    global upscaleModel
    global upscaleDiscModel
    
    upscaleModel.model.load_weights(path+"up/generator")
    upscaleDiscModel.model.load_weights(path+"up/discriminator")
    
def buildBatch(inputImage, numImages, size):
    images = None
    for j in range(numImages):
        posX = random.randrange(0, inputImage.shape[1]-size)
        posY = random.randrange(0, inputImage.shape[0]-size)
        cropped = tf.cast(tf.image.crop_to_bounding_box(inputImage, posY, posX, size, size), dtype=tf.float32) / 255.
        images = tf.expand_dims(cropped, axis=0) if images is None else tf.concat([images,tf.expand_dims(cropped, axis=0)], axis=0)
    return images

def trainOnImage(inputImage, numImages, learningRate):
    trainStepFunction = tf.function(trainStep)
    
    optimizer.learning_rate = learningRate
    discOptimizer.learning_rate = learningRate
    
    for i in range(numImages):
        if i%100 == 0:
            print(i)
        images = None
        #print("epoch %d of 1000"%(i+1))
        images = buildBatch(inputImage, 1, 96)
        #print(images.shape)
        trainStepFunction(images)
            