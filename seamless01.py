import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision

import os
import random
import PIL
import PIL.Image
import threading

from styleloss import *
import time

# uncomment this line to force CPU processing
# tf.config.set_visible_devices([], 'GPU')

# uncomment this line to enable automatic fp16 calculations on NVIDIA GPUs
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

NUM_RESIDUAL_LAYERS = 5
IMAGE_WIDTH_TRAINING = 128
IMAGE_WIDTH_TESTING = 256
DISC_LEARNING_FACTOR = 1.0
NUM_DECODER_CHUNKS = 16
batchSize = 1
files = os.listdir("images")
baseImage = PIL.Image.open("images/albedo.png")
normalBaseImage = PIL.Image.open("images/normal.png")
baseImage = tf.image.crop_to_bounding_box(baseImage, 0,0, baseImage.getbbox()[2], baseImage.getbbox()[3])
normalBaseImage = tf.image.crop_to_bounding_box(normalBaseImage, 0,0, normalBaseImage.getbbox()[2], normalBaseImage.getbbox()[3])
#remove alpha channels and concatenate
baseImage = baseImage[:,:,:3]
normalBaseImage = normalBaseImage[:,:,:3]
baseImage = tf.concat([baseImage, normalBaseImage], axis=2)
normalBaseImage = None

genModel = None
discModel = None
styleModel = createStyleModel()

batchSem = threading.Semaphore()
asyncBatchBuffer = None
asyncBatchBufferCropped = None

class TextureGenerator(Model):
    def __init__(self):
        super(TextureGenerator, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(64, (7,7), strides=2, padding="valid", input_shape=(None, None, 6)),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3,3), strides=2, padding="valid"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, (3,3), strides=1, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU()
            ])
        self.residualLayers = []
        for i in range(NUM_RESIDUAL_LAYERS):
            newResidualLayer = tf.keras.Sequential([
            layers.Conv2D(256, (3,3), strides=1, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU()
            ])
            self.residualLayers.append(newResidualLayer)
        self.albedoDecoder = tf.keras.Sequential([
            layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(64, kernel_size=7, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(3, (3,3), strides=1, padding="same", activation="sigmoid")
            ])
        self.normalDecoder = tf.keras.Sequential([
            layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(64, kernel_size=7, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(3, (3,3), strides=1, padding="same", activation="sigmoid")
            ])
        self.tileLatentSpace = False
        
    def call(self, x, training=False):
        x = tf.pad(x, tf.constant([[0,0],[4, 4,], [4, 4],[0,0]]), "REFLECT")
        result = self.encoder(x)
        if self.tileLatentSpace:
            result = np.tile(result,[1,2,2,1])
        for residualLayer in self.residualLayers:
            result = result + residualLayer(result)
        albedoResult = self.albedoDecoder(result[:,:,:,:3])
        normalResult = self.normalDecoder(result[:,:,:,3:6])
        return tf.concat([albedoResult,normalResult], axis=3)
    def chunkedCall(self, x):
        x = tf.pad(x, tf.constant([[0,0],[4, 4,], [4, 4],[0,0]]), "REFLECT")
        result = self.encoder(x)
        if self.tileLatentSpace:
            result = np.tile(result,[1,2,2,1])
        for residualLayer in self.residualLayers:
            result = result + residualLayer(result)
        
        chunkSize = result.shape[1]//NUM_DECODER_CHUNKS
        result = tf.pad(result, tf.constant([[0,0],[5, 5,], [5, 5],[0,0]]), "REFLECT")
        finalResult = None
        lineResult = None
        for ix in range(NUM_DECODER_CHUNKS):
            for iy in range(NUM_DECODER_CHUNKS):
                chunk = result[:,ix*chunkSize:(ix+1)*chunkSize+10,iy*chunkSize:(iy+1)*chunkSize+10,:]
                chunkResultAlbedo = self.albedoDecoder(chunk[:,:,:,:3])[:,40:-40,40:-40,:]
                if finalResult == None and lineResult == None:
                    print(chunk.shape)
                    print(chunkResultAlbedo.shape)
                chunkResultNormal = self.normalDecoder(chunk[:,:,:,3:6])[:,40:-40,40:-40,:]
                if lineResult == None:
                    lineResult = tf.concat([chunkResultAlbedo, chunkResultNormal], axis=3)
                else:
                    lineResult = tf.concat([lineResult, tf.concat([chunkResultAlbedo, chunkResultNormal], axis=3)], axis=2)
            if finalResult == None:
                finalResult = lineResult
            else:
                finalResult = tf.concat([finalResult, lineResult], axis=1)
            lineResult = None
        return finalResult
    
class TextureDiscriminator(Model):
    def __init__(self):
        super(TextureDiscriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (4,4), strides=2, padding="same", input_shape=(None, None, 6)),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(128, (4,4), strides=1, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(256, (4,4), strides=1, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(512, (4,4), strides=2, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(1, (4,4), strides=1, padding="same", activation="sigmoid"),
            ])
        
    def call(self, x, training=False):
        result = self.model(x)
        return result
        
def discriminatorLoss(realOutput, fakeOutput):
    realLoss = losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(realOutput), realOutput)
    fakeLoss = losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(fakeOutput), fakeOutput)
    return realLoss + fakeLoss;
        
def generatorLoss(fakeOutput, realImages, fakeImages):
    dissLoss = losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(fakeOutput), fakeOutput)
    L1A = losses.MeanAbsoluteError()(realImages[:,:,:,:3], fakeImages[:,:,:,:3])
    L1N = losses.MeanAbsoluteError()(realImages[:,:,:,3:6], fakeImages[:,:,:,3:6])
    styleLoss = calculateStyleLoss(styleModel, realImages[:,:,:,:3], fakeImages[:,:,:,:3])
    #print("%f   -   %f   -   %f" % (dissLoss, likenessLoss, styleLoss))
    return dissLoss + (10.0 * (L1A+L1N)) + styleLoss
        
genOptimizer = optimizers.Adam(0.0002)
discOptimizer = optimizers.Adam(0.0002)

@tf.function
def trainStep(realImages, croppedImages):
    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        fakeImages = genModel(croppedImages, training=True)
        
        realOutput = discModel(realImages, training=True)
        fakeOutput = discModel(fakeImages, training=True)
        
        genLoss = generatorLoss(fakeOutput, realImages, fakeImages)
        discLoss = discriminatorLoss(realOutput, fakeOutput)
        
        genGradients = genTape.gradient(genLoss, genModel.trainable_variables)
        discGradients = discTape.gradient(discLoss, discModel.trainable_variables)
        
        genOptimizer.apply_gradients(zip(genGradients, genModel.trainable_variables))
        discOptimizer.apply_gradients(zip(discGradients, discModel.trainable_variables))


def buildBatch(files, batchSize, maxImageWidth):
    images = []
    #randomFiles = random.choices(files, k=batchSize)
    #for file in randomFiles:
    #    img = PIL.Image.open("images/"+file)
    for i in range(batchSize):
        size = maxImageWidth
        posX = random.randrange(0, baseImage.shape[0]-size)
        posY = random.randrange(0, baseImage.shape[1]-size)
        cropped = tf.image.crop_to_bounding_box(baseImage, posY, posX, size, size)
        images.append(cropped)
    
    baseImages = tf.convert_to_tensor(images).numpy().astype("float32") / 255.
        
    croppedImages = []
    for i in baseImages:
        sx = i.shape[0]//2
        sy = i.shape[1]//2
        ox = sx // 2
        oy = sy // 2
        croppedImages.append(tf.image.crop_to_bounding_box(i, ox,oy,sx,sy))
    
    croppedImages = tf.convert_to_tensor(croppedImages)
    
    return baseImages, croppedImages

def loadTestImage(k):
    images = []
    posX = (baseImage.shape[0] - k)//2
    posY = (baseImage.shape[1] - k)//2
    images.append(tf.image.crop_to_bounding_box(baseImage, posY, posX, k, k))
    
    return tf.convert_to_tensor(images).numpy().astype("float32") / 255.

def saveTestImage(index):
    resultingImage = genModel(testImage[0:1])
    fakeDisOutput = discModel(resultingImage).numpy() * 255.0
    fakeDisOutput = np.tile(fakeDisOutput, (1,1,3))
    realDisOutput = discModel(testImage[0:1]).numpy() * 255.0
    realDisOutput = np.tile(realDisOutput, (1,1,3))
    print(fakeDisOutput.shape)
    print(realDisOutput.shape)
    
    pilImage = PIL.Image.fromarray((resultingImage[0,:,:,:3].numpy() * 255.0).astype("uint8"))
    pilImageNormal = PIL.Image.fromarray((resultingImage[0,:,:,3:6].numpy() * 255.0).astype("uint8"))
    pil2Image = PIL.Image.fromarray((testImage[0,:,:,:3]*255.).astype("uint8"))
    pil2ImageNormal = PIL.Image.fromarray((testImage[0,:,:,3:6]*255.).astype("uint8"))
    pilDisImage = PIL.Image.fromarray(fakeDisOutput[0].astype("uint8"))
    pilDisImage2 = PIL.Image.fromarray(realDisOutput[0].astype("uint8"))
    outputImage = PIL.Image.new("RGB", (IMAGE_WIDTH_TESTING * 4, IMAGE_WIDTH_TESTING * 4))
    outputImage.paste(pilImage, (0,0))
    outputImage.paste(pilImageNormal, (0,IMAGE_WIDTH_TESTING * 2))
    outputImage.paste(pil2Image, (IMAGE_WIDTH_TESTING * 2,0))
    outputImage.paste(pil2ImageNormal, (IMAGE_WIDTH_TESTING * 2,IMAGE_WIDTH_TESTING * 2))
    outputImage.paste(pilDisImage, (IMAGE_WIDTH_TESTING * 2, IMAGE_WIDTH_TESTING))
    outputImage.paste(pilDisImage2, (IMAGE_WIDTH_TESTING * 2, IMAGE_WIDTH_TESTING+IMAGE_WIDTH_TESTING//2))
    
    outputImage.save("results/"+str(index)+".jpg")

def asyncLoadBatch(files, batchSize, k):
    global batchSem
    global asyncBatchBuffer
    global asyncBatchBufferCropped
    batchSem.acquire()
    #print("ding!")
    asyncBatchBuffer, asyncBatchBufferCropped = buildBatch(files, batchSize, k * 2)
    batchSem.release()

def train(startI, untilI, learningRate=0.0002, k=IMAGE_WIDTH_TRAINING, imageEveryXBatches = 100):
    global batchSem
    global asyncBatchBuffer
    global asyncBatchBufferCropped
    iterations = untilI - startI
    genOptimizer.learning_rate = learningRate
    discOptimizer.learning_rate = learningRate * DISC_LEARNING_FACTOR
    
    threading.Thread(target=asyncLoadBatch, args=(files,batchSize,k,)).start()
    
    startTime = time.time()
    
    for i in range(startI, startI+iterations):
        if i%100 == 0:
            if i-startI != 0:
                elapsedTime = time.time() - startTime
                timeLeftFactor = 1 / (i-startI) * (iterations - (i - startI)) / 60.0
                print("working on batch %d, minutes remaining for this step: %d" % (i, int(elapsedTime * timeLeftFactor)))
            else:
                print("working on batch %d" % (i))
        batchSem.acquire()
        baseImages = tf.Variable(asyncBatchBuffer)
        croppedImages = tf.Variable(asyncBatchBufferCropped)
        threading.Thread(target=asyncLoadBatch, args=(files,batchSize,k,)).start()
        batchSem.release()
        trainStep(baseImages, croppedImages)
        if i%imageEveryXBatches == 0:
            saveTestImage(i)
            
def saveTileableTextures(k):
    
    genModel.tileLatentSpace = True
    genInput = loadTestImage(k)
    genOutput = genModel.chunkedCall(genInput)
    
    albedoMap = PIL.Image.fromarray((genOutput[0,k:3*k,k:3*k,:3].numpy() * 255.0).astype("uint8"))
    normalMap = PIL.Image.fromarray((genOutput[0,k:3*k,k:3*k,3:6].numpy() * 255.0).astype("uint8"))
    
    albedoMap.save("results/albedo.png")
    normalMap.save("results/normal.png")
    
    genModel.tileLatentSpace = False
            
def createModels():
    global genModel
    global discModel
    genModel = TextureGenerator()
    discModel = TextureDiscriminator()

def loadModels():
    global genModel
    global discModel
    createModels()
    genModel.load_weights("gen/weights")
    discModel.load_weights("disc/weights")
    
def saveModels():
    global genModel
    global discModel
    genModel.save_weights("gen/weights")
    discModel.save_weights("disc/weights")

def stdLearning():
    sTime = time.time()
    createModels()
    tf.keras.backend.clear_session()
    train(0,15000,0.0002,IMAGE_WIDTH_TRAINING,500)
    saveModels()
    train(15000,30000,0.0002,IMAGE_WIDTH_TRAINING,500)
    saveModels()
    train(30000,40000,0.00004,IMAGE_WIDTH_TRAINING,500)
    saveModels()
    train(40000,45000,0.00001,IMAGE_WIDTH_TRAINING,500)
    saveModels()
    train(45000,50001,0.000008,IMAGE_WIDTH_TRAINING,500)
    saveModels()
    tf.keras.backend.clear_session()
    saveTileableTextures(1024)
    print("finished training in %d minutes. Bye!" % int((time.time()-sTime)/60))
    
#experimental method for learning speedup evaluation
def turboLearning():
    sTime = time.time()
    createModels()
    tf.keras.backend.clear_session()
    train(0,1000,0.00075,IMAGE_WIDTH_TRAINING,100)
    train(1000,2000,0.000333,IMAGE_WIDTH_TRAINING,100)
    train(2000,3000,0.0001,IMAGE_WIDTH_TRAINING,100)
    train(3000,4000,0.00001,IMAGE_WIDTH_TRAINING,100)
    train(4000,5000,0.000008,IMAGE_WIDTH_TRAINING,100)
    saveModels()
    tf.keras.backend.clear_session()
    saveTileableTextures(1024)
    print("finished turbo training in %d minutes. Bye!" % int((time.time()-sTime)/60))

testImage = loadTestImage(256)
print("initialized!")