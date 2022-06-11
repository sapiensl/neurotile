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

from styleloss import *
import time

# uncomment this line to force CPU processing
# tf.config.set_visible_devices([], 'GPU')

# uncomment this line to enable automatic fp16 calculations on NVIDIA GPUs
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

USE_L1 = True
USE_LADV = True
USE_LSTYLE = True

USE_PATCH_L1 = False

SILENT = False

NUM_RESIDUAL_BLOCKS = 5
IMAGE_WIDTH_TRAINING = 128
IMAGE_WIDTH_TESTING = 256
DISC_LEARNING_FACTOR = 1.0
NUM_DECODER_CHUNKS = 8
batchSize = 1

genModel = None
discModel = None
styleModel = createStyleModel()

batchSem = threading.Semaphore()
asyncBatchBuffer = None
asyncBatchBufferCropped = None

lastL1 = 0.0
lastLStyle = 0.0
lastLAdv = 0.0

class TextureGenerator(Model):
    def __init__(self, numTextures=2):
        super(TextureGenerator, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(64, (7,7), strides=2, padding="valid", input_shape=(None, None, numTextures*3)),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3,3), strides=2, padding="valid"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, (3,3), strides=1, padding="valid"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU()
            ])
        self.residualBlocks = []
        for i in range(NUM_RESIDUAL_BLOCKS):
            newResidualBlock = tf.keras.Sequential([
            layers.Conv2D(256, (3,3), strides=1, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU()
            ])
            self.residualBlocks.append(newResidualBlock)
        self.decoders = []
        for i in range(numTextures):
            self.decoders.append(tf.keras.Sequential([
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
            )
        self.tileLatentSpace = False
        self.numTextures = numTextures
        
    def call(self, x, training=False):
        x = tf.pad(x, tf.constant([[0,0],[8, 8,], [8, 8],[0,0]]), "REFLECT")
        result = self.encoder(x)
        if self.tileLatentSpace:
            result = np.tile(result,[1,2,2,1])
        for residualBlock in self.residualBlocks:
            result = result + residualBlock(result)#tf.pad(residualBlock(result), tf.constant([[0,0],[1,1],[1,1],[0,0]]))
        concatenatedResult = None
        for i,decoder in enumerate(self.decoders):
            nthTexture = decoder(result)
            concatenatedResult = nthTexture if concatenatedResult == None else tf.concat([concatenatedResult,nthTexture], axis=3)
        return concatenatedResult
    def chunkedCall(self, x):
        global NUM_DECODER_CHUNKS
        x = tf.pad(x, tf.constant([[0,0],[8, 8,], [8, 8],[0,0]]), "REFLECT")
        result = self.encoder(x)
        if self.tileLatentSpace:
            result = np.tile(result,[1,2,2,1])
        for residualBlock in self.residualBlocks:
            result = result + residualBlock(result)#tf.pad(residualBlock(result), tf.constant([[0,0],[1,1],[1,1],[0,0]]))
        
        chunkSize = result.shape[1]//NUM_DECODER_CHUNKS
        result = tf.pad(result, tf.constant([[0,0],[5, 5,], [5, 5],[0,0]]), "REFLECT")
        finalResult = None
        lineResult = None
        for ix in range(NUM_DECODER_CHUNKS):
            for iy in range(NUM_DECODER_CHUNKS):
                chunk = result[:,ix*chunkSize:(ix+1)*chunkSize+10,iy*chunkSize:(iy+1)*chunkSize+10,:]
                concatenatedChunkResult = None
                for i,decoder in enumerate(self.decoders):
                    nthTexture = decoder(chunk)
                    concatenatedChunkResult = nthTexture if concatenatedChunkResult == None else tf.concat([concatenatedChunkResult,nthTexture], axis=3)
                if lineResult == None:
                    lineResult = concatenatedChunkResult
                else:
                    lineResult = tf.concat([lineResult, concatenatedChunkResult], axis=2)
            if finalResult == None:
                finalResult = lineResult
            else:
                finalResult = tf.concat([finalResult, lineResult], axis=1)
            lineResult = None
        return finalResult
    
class TextureDiscriminator(Model):
    def __init__(self, numTextures):
        super(TextureDiscriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (4,4), strides=2, padding="same", input_shape=(None, None, numTextures*3)),
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

def patchL1(in1, in2):
    outSize = in1.shape[1]//4
    diff = tf.abs(in1-in2)
    outTensor = None
    outTensorLine = []
    for iy in range(outSize):
        for ix in range(outSize):
            outTensorLine.append(tf.reduce_mean(diff[:,ix*4:(ix+1)*4, iy*4:(iy+1)*4,:]))
        if outTensor == None:
            outTensor = tf.expand_dims(outTensorLine, axis=1)
        else:
            outTensor = tf.concat([outTensor, tf.expand_dims(tf.stack(outTensorLine), axis=1)], axis=1)
        outTensorLine = []
        
    return tf.expand_dims(tf.expand_dims(outTensor, 0), 3)
        
def generatorLoss(fakeOutput, realImages, fakeImages):
    global lastL1
    global lastLAdv
    global lastLStyle
    
    if USE_LADV:
        discLoss = losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(fakeOutput), fakeOutput)
        lastLAdv = tf.math.reduce_sum(discLoss)
    else:
        discLoss = 0
        lastAdv = 0.0
    if USE_L1:
        if USE_PATCH_L1:
            L1 = patchL1(realImages, fakeImages)
            lastL1 = tf.math.reduce_sum(L1)
        else:
            L1 = losses.MeanAbsoluteError()(realImages[:,:,:,:3], fakeImages[:,:,:,:3])
            lastL1 = L1
    else:
        L1 = 0
        lastL1 = 0.0
    
    styleLoss = 0
    lastLStyle = 0.0
    if USE_LSTYLE:
        for i in range(genModel.numTextures):
            styleLoss += calculateStyleLoss(styleModel, realImages[:,:,:,i*3:(i+1)*3], fakeImages[:,:,:,i*3:(i+1)*3])
        lastLStyle = styleLoss
    #print("%f   -   %f   -   %f" % (dissLoss, likenessLoss, styleLoss))
    return discLoss + (10.0 * L1) + styleLoss
        
genOptimizer = optimizers.Adam(0.0002)
discOptimizer = optimizers.Adam(0.0002)

def trainStep(realImages, croppedImages):
    print("trainStep(...) was retraced!")
    
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
    global baseImage
    if baseImage == None:
        loadImageStack()
    images = None
    #randomFiles = random.choices(files, k=batchSize)
    #for file in randomFiles:
    #    img = PIL.Image.open("images/"+file)
    for i in range(batchSize):
        size = maxImageWidth
        posX = random.randrange(0, baseImage.shape[1]-size)
        posY = random.randrange(0, baseImage.shape[0]-size)
        cropped = tf.cast(tf.image.crop_to_bounding_box(baseImage, posY, posX, size, size), dtype=tf.float32) / 255.
        images = tf.expand_dims(cropped, axis=0) if images == None else tf.stack([images,cropped])
        
    croppedImages = None
    for i in images:
        sx = i.shape[0]//2
        sy = i.shape[1]//2
        ox = sx // 2
        oy = sy // 2
        subCrop = tf.cast(tf.image.crop_to_bounding_box(i,ox,oy,sx,sy), dtype=tf.float32) / 255.
        croppedImages = tf.expand_dims(subCrop, axis=0) if croppedImages == None else tf.stack([croppedImages,subCrop])
        
    return images, croppedImages

def loadTestImage(k):
    global baseImage
    if baseImage == None:
        loadImageStack()
    images = []
    posX = (baseImage.shape[0] - k)//2
    posY = (baseImage.shape[1] - k)//2
    images.append(tf.image.crop_to_bounding_box(baseImage, posY, posX, k, k))
    
    return tf.convert_to_tensor(images).numpy().astype("float32") / 255.

def saveTestImage(index):
    global testImage
    
    if testImage is None:
        testImage = loadTestImage(IMAGE_WIDTH_TESTING)
    
    resultingImage = genModel(testImage[0:1])
    fakeDisOutput = discModel(resultingImage).numpy() * 255.0
    fakeDisOutput = np.tile(fakeDisOutput, (1,1,3))
    realDisOutput = discModel(testImage[0:1]).numpy() * 255.0
    realDisOutput = np.tile(realDisOutput, (1,1,3))
    print(fakeDisOutput.shape)
    print(realDisOutput.shape)
    
    
    outputImage = PIL.Image.new("RGB", (IMAGE_WIDTH_TESTING * 3, IMAGE_WIDTH_TESTING * 2 * genModel.numTextures))

    #print all subtextures to the output image
    for i in range(genModel.numTextures):
        subImage = PIL.Image.fromarray((resultingImage[0,:,:,i*3:(i+1)*3].numpy() * 255.0).astype("uint8"))
        subInput = PIL.Image.fromarray((testImage[0,:,:,i*3:(i+1)*3]*255.).astype("uint8"))
        outputImage.paste(subImage, (0, IMAGE_WIDTH_TESTING * 2 * i))
        outputImage.paste(subInput, (IMAGE_WIDTH_TESTING * 2, IMAGE_WIDTH_TESTING * 2 * i))

    pilDisImage = PIL.Image.fromarray(fakeDisOutput[0].astype("uint8"))
    pilDisImage2 = PIL.Image.fromarray(realDisOutput[0].astype("uint8"))
    outputImage.paste(pilDisImage, (IMAGE_WIDTH_TESTING * 2, IMAGE_WIDTH_TESTING))
    outputImage.paste(pilDisImage2, (IMAGE_WIDTH_TESTING * 2, IMAGE_WIDTH_TESTING+IMAGE_WIDTH_TESTING//2))
    
    outputImage.save(currentProjectPath+str(index)+".jpg")
    
def clearLossLog():
    with open(currentProjectPath + "losses.csv", "w") as lossCSV:
        lossCSV.write("LAdv;L1;LStyle;\n")

def logLossValues(i): 
    realImages, croppedImages = buildBatch(None, 1, IMAGE_WIDTH_TRAINING*2)
    fakeImages = genModel(croppedImages, training=False)
    fakeOutput = discModel(fakeImages, training=False)
    _ = generatorLoss(fakeOutput, realImages, fakeImages)
    global lastLAdv
    global lastL1
    global lastLStyle
    with open(currentProjectPath + "losses.csv", "a") as lossCSV:
        lossCSV.write("%d;%f;%f;%f\n" % (i, lastLAdv, lastL1, lastLStyle))
        print("i: %d\tLAdv: %f\tL1: %f\tLStyle: %f" % (i, lastLAdv, lastL1, lastLStyle))
        
def plotLosses():
    iterations = []
    LAdv = []
    L1 = []
    LStyle = []
    
    firstRow = True
    try:
        with open(currentProjectPath + "losses.csv", "r") as lossCSV:
            lossReader = csv.reader(lossCSV, delimiter=";")
            for i, row in enumerate(lossReader):
                if firstRow:
                    firstRow = False
                    continue
                iterations.append(float(row[0]))
                LAdv.append(float(row[1]))
                L1.append(float(row[2]))
                LStyle.append(float(row[3]))
                
        figure, axis = plt.subplots(3)
        axis[0].plot(iterations,LAdv)
        axis[0].set_title("LAdv")
        axis[1].plot(iterations,L1)
        axis[1].set_title("L1")
        axis[2].plot(iterations, LStyle)
        axis[2].set_title("LStyle")
          
        plt.show()
    except:
        print(traceback.format_exc())
        print("could not open losses.csv in this projects directory.")

def asyncLoadBatch(files, batchSize, k):
    global batchSem
    global asyncBatchBuffer
    global asyncBatchBufferCropped
    batchSem.acquire()
    #print("ding!")
    asyncBatchBuffer, asyncBatchBufferCropped = buildBatch(files, batchSize, k * 2)
    batchSem.release()

def train(startI, untilI, learningRate=0.0002, k=IMAGE_WIDTH_TRAINING, imageEveryXBatches = 100, lossStatsEveryXBatches=100):
    global batchSem
    global asyncBatchBuffer
    global asyncBatchBufferCropped
    global SILENT
    global genModel
    
    iterations = untilI - startI
    genOptimizer.learning_rate = learningRate
    discOptimizer.learning_rate = learningRate * DISC_LEARNING_FACTOR
    
    trainStepFunction = tf.function(trainStep, input_signature=(tf.TensorSpec(shape=[batchSize,k*2,k*2,genModel.numTextures*3], dtype=tf.float32),tf.TensorSpec(shape=[batchSize,k,k,genModel.numTextures*3], dtype=tf.float32),))
    
    threading.Thread(target=asyncLoadBatch, args=(files,batchSize,k,)).start()
    
    startTime = time.time()
    
    for i in range(startI, startI+iterations):
        if SILENT:
            time.sleep(0.5)
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
        trainStepFunction(baseImages, croppedImages)
        if i%imageEveryXBatches == 0:
            saveTestImage(i)
        if i%lossStatsEveryXBatches == 0:
            logLossValues(i)

def saveTileableTextures(k, crop=True, filesuffix="", customInput=None):
    global genModel
    
    genModel.tileLatentSpace = True
    if customInput != None:
        genInput = customInput
        k = customInput.shape[1]
    else:
        genInput = loadTestImage(k)
        
    #we assume that the tileable texture has a pretty big input and use the chunked call by default.
    genOutput = genModel.chunkedCall(genInput)
    
    for i in range(genModel.numTextures):
        if crop:
            texture = PIL.Image.fromarray((genOutput[0,k:3*k,k:3*k,i*3:(i+1)*3].numpy() * 255.0).astype("uint8"))
        else:
            texture = PIL.Image.fromarray((genOutput[0,:,:,i*3:(i+1)*3].numpy() * 255.0).astype("uint8"))
        
        texture.save(("%soutput%d%s.png" % (currentProjectPath, i, (("_" + filesuffix) if filesuffix else ""))))
    
    genModel.tileLatentSpace = False

def saveGeneratorWeights(path):
    global genModel
    global discModel
    
    genModel.encoder.save_weights(path+"gen/enc")
    for i, resBlock in enumerate(genModel.residualBlocks):
        resBlock.save_weights(path+"gen/res"+str(i))
    for i, decoder in enumerate(genModel.decoders):
        decoder.save_weights(path+("gen/dec%d" % i))

def saveDiscriminatorWeights(path):
    discModel.model.save_weights(path+"disc/weights")
    
def loadWeights(path):
    global genModel
    global discModel
    
    genModel.encoder.load_weights(path+"gen/enc")
    for i, resBlock in enumerate(genModel.residualBlocks):
        resBlock.load_weights(path+"gen/res"+str(i))
    for i, decoder in enumerate(genModel.decoders):
        decoder.load_weights(path+("gen/dec%d" % i))
    
    discModel.model.load_weights(path+"disc/weights")

def createModels():
    global genModel
    global discModel
    global baseImage
    
    #we need to make sure the input image stack is loaded to be able to tell the number of channels the generator model needs
    if baseImage == None:
        loadImageStack()
    
    #baseImage does not have the leading axis yet, so the number of channels are stored in axis index 2
    numberOfTextures = baseImage.shape[2] // 3
    
    #if genModel == None:
    genModel = TextureGenerator(numTextures=numberOfTextures)
    genModel(tf.fill([1,128,128,genModel.numTextures*3],0.0))
        #saveGeneratorWeights("")
    #if discModel == None:
    discModel = TextureDiscriminator(numTextures=numberOfTextures)
        #saveDiscriminatorWeights("")
    #loadWeights("")
    
def loadModels():
    global genModel
    global discModel
    createModels()
    loadWeights(currentProjectPath)
    
def saveModels():
    global genModel
    global discModel
    saveGeneratorWeights(currentProjectPath)
    saveDiscriminatorWeights(currentProjectPath)
    
def setProject(projectName):
    global baseImage
    global testImage
    global currentProjectPath
    baseImage = None
    testImage = None
    oldProjectPath = currentProjectPath
    currentProjectPath = "projects/"+projectName+"/"
    try:
        os.mkdir(currentProjectPath)
    except:
        pass
    try:
        os.mkdir(currentProjectPath+"images")
        #shutil.copyfile(oldProjectPath + "images/albedo.png", currentProjectPath + "images/albedo.png")
        #shutil.copyfile(oldProjectPath + "images/normal.png", currentProjectPath + "images/normal.png")
    except:
        pass

def loadImageStack():
    global baseImage
    baseImage = None
    
    for imgFilename in os.listdir(currentProjectPath+"images/"):
        if os.path.isfile(currentProjectPath+"images/"+imgFilename):
            try:
                image = PIL.Image.open(currentProjectPath+"images/"+imgFilename)
                #convert to tensor (?) TODO: check if there is a cleaner option
                image = tf.image.crop_to_bounding_box(image, 0,0, image.getbbox()[3], image.getbbox()[2])
                #remove alpha channel
                image = image[:,:,:3]
                #add to stack
                baseImage = image if baseImage == None else tf.concat([baseImage,image], axis=2)
            except:
                print("Warning: could not load base image with name %s, skipping..." % imgFilename)
    
    
def saveImage(inputData, filename):
    PIL.Image.fromarray((inputData.numpy() * 255.0).astype("uint8")).save(currentProjectPath+filename+".png")
    
def stdLearning():
    sTime = time.time()
    createModels()
    clearLossLog()
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
    clearLossLog()
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

def ablationTest():
    global USE_L1
    global USE_LSTYLE
    global USE_LADV
    
    setProject("l1")
    createModels()
    USE_L1 = True
    USE_LADV = False
    USE_LSTYLE = False
    stdLearning()
    saveModels()
    setProject("lstyle")
    createModels()
    USE_L1 = False
    USE_LADV = False
    USE_LSTYLE = True
    stdLearning()
    saveModels()
    setProject("ladv")
    createModels()
    USE_L1 = False
    USE_LADV = True
    USE_LSTYLE = False
    stdLearning()
    saveModels()
    USE_L1 = True
    USE_LADV = True
    USE_LSTYLE = True
    
def noiseExperiment(startK=16, iterations=5, makeFinalStack=True):
    global currentProjectPath
    originalPath = currentProjectPath
    currentProjectPath += "fromnoise/"
    os.mkdir(currentProjectPath)
    inputNoise = tf.random.uniform(shape=(1,startK,startK,genModel.numTextures*3), minval=0.0, maxval=1.0)
    saveImage(inputNoise[0,:,:,:3], ("n%d" % startK))
    output = None
    iterK = startK
    for i in range(iterations):
        if i == 0:
            output = genModel(inputNoise)
        else:
            if iterK > 200:
                output = genModel.chunkedCall(output)
            else:
                output = genModel(output)
        iterK *= 2
        saveImage(output[0,:,:,:3], ("n%d" % iterK))
    if makeFinalStack:
        saveTileableTextures(0, True, "_fromnoise", output)
    currentProjectPath = originalPath
    
def noiseSweep(k=256):
    info = tf.Variable(loadTestImage(k))#TODO fix output of loadTestImage to correct type "TF Tensor"
    global currentProjectPath
    originalPath = currentProjectPath
    currentProjectPath += "noisesweep/"
    try:
        os.mkdir(currentProjectPath)
    except:
        #I really dont care right now, this is experimental code executed by real professionals
        pass

    noise = tf.random.uniform(shape=(1,k,k,genModel.numTextures*3), minval=0.05, maxval=0.95)
    
    for i in range(101):
        infoFactor = 1 - (i/100)
        noiseFactor = i/100
        mix = infoFactor * info + noiseFactor * noise
        saveImage(mix[0,:,:,:3], ("mix%d" % i))
        saveTileableTextures(0, True, ("noise%dpercent" %i), mix)
    currentProjectPath = originalPath

currentProjectPath = "projects/default/"
baseImage = loadImageStack()
files = os.listdir(currentProjectPath+"images")
setProject("default")
testImage = loadTestImage(256)
print("initialized!")