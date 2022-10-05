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

from styleloss import *
import time

# uncomment this line to force CPU processing
# tf.config.set_visible_devices([], 'GPU')

# uncomment this line to enable automatic fp16 calculations on NVIDIA GPUs
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

USE_L1 = True
USE_LADV = True
USE_LSTYLE = True

LAMBDA_L1 = 10.0
LAMBDA_LADV = 1.0
LAMBDA_LSTYLE = 1.0

#these two decide how many times the resulting image is downsampled and sent through the respective loss calculation
#1 means that only the original output will be taken into account
#2 means that the original and a 1/2x downsampled version of it will be processed and the losses will be added together
#3 means original, 1/2x and 1/4x are processed
# and so on and so forth
LSTYLE_LAYERS = 1
LADV_LAYERS = 1 #TODO:implement

USE_PATCH_L1 = False #TODO: might be broken by now, has not been maintained in a good while

SILENT = False

FORCE_INPUT_RESOLUTION = False
USE_UPSCALER = False


NUM_RESIDUAL_BLOCKS = 5
KERNEL_SIZE_RESIDUAL_BLOCKS = 3
IMAGE_WIDTH_TRAINING = 128
IMAGE_WIDTH_TESTING = 256
IMAGE_USE_ADVANCED_PADDING = True # use the image itself to pad the input (during training) instead of reflection padding
DISC_LEARNING_FACTOR = 1.0
NUM_DECODER_CHUNKS = 8
ENCODER_DEPTH = 2 #meaning the input texture will be (1/(2^ENCODER_DEPTH)) in size when encoded, standard is 2 -> 1/4

FEATUREMAP_START_SIZE = 64
FEATUREMAP_INCREASE_FAC = 2
FEATUREMAP_MAX_SIZE = 8192

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
        
        currentFMapSize = FEATUREMAP_START_SIZE
        
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(currentFMapSize, (7,7), strides=2, padding="valid", input_shape=(None, None, numTextures*3)))
        self.encoder.add(tfa.layers.InstanceNormalization())
        self.encoder.add(layers.ReLU())
        
        for i in range(1,ENCODER_DEPTH):
            currentFMapSize = min(currentFMapSize * FEATUREMAP_INCREASE_FAC, FEATUREMAP_MAX_SIZE)
            self.encoder.add(layers.Conv2D(currentFMapSize, (3,3), strides=2, padding="valid"))
            self.encoder.add(tfa.layers.InstanceNormalization())
            self.encoder.add(layers.ReLU())
            
        currentFMapSize = min(currentFMapSize * FEATUREMAP_INCREASE_FAC, FEATUREMAP_MAX_SIZE)
        
        self.encoder.add(layers.Conv2D(currentFMapSize, (3,3), strides=1, padding="valid"))
        self.encoder.add(tfa.layers.InstanceNormalization())
        self.encoder.add(layers.ReLU())
                         
        self.residualBlocks = []
        for i in range(NUM_RESIDUAL_BLOCKS):
            newResidualBlock = tf.keras.Sequential([
            layers.Conv2D(currentFMapSize, (KERNEL_SIZE_RESIDUAL_BLOCKS,KERNEL_SIZE_RESIDUAL_BLOCKS), strides=1, padding="same"),
            tfa.layers.InstanceNormalization(),
            layers.ReLU()
            ])
            self.residualBlocks.append(newResidualBlock)
        
        self.decoders = []
        for i in range(numTextures):
            currentDecoderFMapSize = currentFMapSize
            newDecoder = tf.keras.Sequential()
            newDecoder.add(layers.Conv2DTranspose(currentDecoderFMapSize, kernel_size=3, strides=2, padding="same"))
            newDecoder.add(tfa.layers.InstanceNormalization())
            newDecoder.add(layers.ReLU())
            
            for j in range(1, ENCODER_DEPTH):
                currentDecoderFMapSize //= FEATUREMAP_INCREASE_FAC
                newDecoder.add(layers.Conv2DTranspose(currentDecoderFMapSize, kernel_size=3, strides=2, padding="same"))
                newDecoder.add(tfa.layers.InstanceNormalization())
                newDecoder.add(layers.ReLU())

            currentDecoderFMapSize //= FEATUREMAP_INCREASE_FAC
            newDecoder.add(layers.Conv2DTranspose(currentDecoderFMapSize, kernel_size=7, strides=2, padding="same"))
            newDecoder.add(tfa.layers.InstanceNormalization())
            newDecoder.add(layers.ReLU())
            newDecoder.add(layers.Conv2D(3, (3,3), strides=1, padding="same", activation="sigmoid"))
            self.decoders.append(newDecoder)
        self.tileLatentSpace = False
        self.tileMirrored = False
        self.numTextures = numTextures
        self.requiredPadding = 2
        for i in range(1,ENCODER_DEPTH):
            self.requiredPadding += 2**(i+1)
        self.requiredPadding += 3
        #print("requiredPadding for encoding depth %d is %d" % (ENCODER_DEPTH, self.requiredPadding))
        
    def call(self, x, training=False):
        p = self.requiredPadding
        #only pad out the image if the input is not already padded
        if not (training and IMAGE_USE_ADVANCED_PADDING):
            x = tf.pad(x, tf.constant([[0,0],[p, p], [p, p],[0,0]]), "REFLECT")
        result = self.encoder(x)
        if self.tileLatentSpace:
            if self.tileMirrored:
                result = tf.concat([result,tf.reverse(result,axis=[1])],axis=1)
                result = tf.concat([result,tf.reverse(result,axis=[2])],axis=2)
            else:
                result = tf.tile(result,[1,2,2,1])
        for residualBlock in self.residualBlocks:
            result = result + residualBlock(result)#tf.pad(residualBlock(result), tf.constant([[0,0],[1,1],[1,1],[0,0]]))
        concatenatedResult = None
        for i,decoder in enumerate(self.decoders):
            nthTexture = decoder(result)
            concatenatedResult = nthTexture if concatenatedResult == None else tf.concat([concatenatedResult,nthTexture], axis=3)
        return concatenatedResult
    def chunkedCall(self, x):
        global NUM_DECODER_CHUNKS
        p = self.requiredPadding
        x = tf.pad(x, tf.constant([[0,0],[p, p,], [p, p],[0,0]]), "REFLECT")
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
                    concatenatedChunkResult = nthTexture[:,40:-40,40:-40,:] if concatenatedChunkResult == None else tf.concat([concatenatedChunkResult,nthTexture[:,40:-40,40:-40,:]], axis=3)
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
    global LAMBDA_L1
    global LAMBDA_LADV
    global LAMBDA_STYLE
    global LSTYLE_LAYERS
    
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
            #print(realImages.shape)
            #print(fakeImages.shape)
            L1 = losses.MeanAbsoluteError()(realImages, fakeImages)
            lastL1 = L1
    else:
        L1 = 0
        lastL1 = 0.0
    
    styleLoss = 0
    lastLStyle = 0.0
    if USE_LSTYLE:
        for i in range(genModel.numTextures):
            lstyleprefactor = 1.0
            for downsamplingI in range(LSTYLE_LAYERS):
                sampleSize = realImages.shape[1]//(2**downsamplingI)
                realImg = realImages[:,:,:,i*3:(i+1)*3] if downsamplingI == 0 else tf.image.resize(realImages[:,:,:,i*3:(i+1)*3],(sampleSize, sampleSize))
                fakeImg = fakeImages[:,:,:,i*3:(i+1)*3] if downsamplingI == 0 else tf.image.resize(fakeImages[:,:,:,i*3:(i+1)*3],(sampleSize, sampleSize))
                individualStyleLoss = lstyleprefactor * calculateStyleLoss(styleModel, realImg, fakeImg)
                styleLoss += (1/(genModel.numTextures*LSTYLE_LAYERS)) * individualStyleLoss
                lstyleprefactor /= 100.0
                
        lastLStyle = styleLoss
    #print("%f   -   %f   -   %f" % (dissLoss, likenessLoss, styleLoss))
    return (LAMBDA_LADV * discLoss) + (LAMBDA_L1 * L1) + (LAMBDA_LSTYLE * styleLoss)
        
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

def getSubImage(x,y,k):
    images = None
    cropped = tf.cast(tf.image.crop_to_bounding_box(baseImage, y, x, k, k), dtype=tf.float32) / 255.
    cropped = tf.expand_dims(cropped, axis=0)
    return cropped

def buildBatch(files, batchSize, maxImageWidth, isTrain=False):
    global baseImage
    global genModel
    
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
        
        if genModel is not None:
            padding = genModel.requiredPadding if (isTrain and IMAGE_USE_ADVANCED_PADDING) else 0
        else:
            padding = 0
            
        subCrop = tf.image.crop_to_bounding_box(i, ox - padding, oy - padding, sx + 2*padding, sy + 2*padding)
        #subCrop = subCrop + tf.random.uniform(shape=subCrop.shape, minval= 0.0, maxval = 0.025)
        croppedImages = tf.expand_dims(subCrop, axis=0) if croppedImages == None else tf.stack([croppedImages,subCrop])
        
    return images, croppedImages

def loadTestImage(k):
    global baseImage
    if baseImage == None:
        loadImageStack()
    images = []
    posX = (baseImage.shape[1] - k)//2
    posY = (baseImage.shape[0] - k)//2
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
    genLoss = generatorLoss(fakeOutput, realImages, fakeImages)
    global lastLAdv
    global lastL1
    global lastLStyle
    with open(currentProjectPath + "losses.csv", "a") as lossCSV:
        lossCSV.write("%d;%f;%f;%f\n" % (i, lastLAdv, lastL1, lastLStyle))
        print("i: %d\tLAdv: %f\tL1: %f\tLStyle: %f" % (i, lastLAdv, lastL1, lastLStyle))

def plotLosses(startIndex=0, saveAsPDF=False):
    iterations = []
    LAdv = []
    L1 = []
    LStyle = []
    LAdvSmooth = []
    LStyleSmooth = []
    L1Smooth = []
    
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
                
        LAdvSmooth = [sum(LAdv[i:i+10])/10.0 for i in range(len(LAdv)-10)]
        L1Smooth = [sum(L1[i:i+10])/10.0 for i in range(len(L1)-10)]
        LStyleSmooth = [sum(LStyle[i:i+10])/10.0 for i in range(len(LStyle)-10)]
        
        figure, axis = plt.subplots(3, figsize=(8,10), dpi=80)
        figure.canvas.manager.set_window_title(currentProjectPath)
        axis[0].plot(iterations[startIndex:],L1[startIndex:], color="blue")
        axis[0].plot(iterations[startIndex:-10],L1Smooth[startIndex:], color="red")
        axis[0].set_title(currentProjectPath+"  -  L1 -> %f" % min(L1Smooth))
        axis[0].grid(True)
        axis[1].plot(iterations[startIndex:],LAdv[startIndex:], color="blue")
        axis[1].plot(iterations[startIndex:-10],LAdvSmooth[startIndex:], color="red")
        axis[1].set_title("LAdv -> %f" % min(LAdvSmooth))
        axis[1].grid(True)
        axis[2].plot(iterations[startIndex:], LStyle[startIndex:], color="blue")
        axis[2].plot(iterations[startIndex:-10],LStyleSmooth[startIndex:], color="red")
        axis[2].set_title("LStyle -> %f" % min(LStyleSmooth))
        axis[2].grid(True)
        plt.tight_layout()
        
        if saveAsPDF:
            plt.savefig(currentProjectPath+"losses.pdf")
            plt.close()
        else:
            plt.show(block=False)
            
    except:
        print(traceback.format_exc())
        print("could not open losses.csv in this projects directory.")
        
def saveAllLossPDFs():
    global currentProjectPath
    global currentProjectFolder
    pathBackup = currentProjectPath
    for project in os.listdir(currentProjectFolder):
        print(project)
        setProject(project)
        plotLosses(10,True)
    currentProjectPath = pathBackup
    
def copyAllLossPDFs():
    global currentProjectFolder
    for project in os.listdir(currentProjectFolder):
        try:
            shutil.copyfile(currentProjectFolder+project+"/losses.pdf", currentProjectFolder+project+".pdf")
        except:
            pass
        
def saveAllTileableTextures(inputSize):
    global currentProjectPath
    global currentProjectFolder
    pathBackup = currentProjectPath
    for project in os.listdir(currentProjectFolder):
        print(project)
        setProject(project)
        try:
            loadModels()
            saveTileableTextures(inputSize)
        except:
            pass
    currentProjectPath = pathBackup
    try:
        loadModels()
    except:
        pass
    
def copyAllTileableTextures():
    global currentProjectFolder
    for project in os.listdir(currentProjectFolder):
        try:
            shutil.copyfile(currentProjectFolder+project+"/output0.png", currentProjectFolder+project+".png")
        except:
            pass

def asyncLoadBatch(files, batchSize, k, isTrain=False):
    global batchSem
    global asyncBatchBuffer
    global asyncBatchBufferCropped
    batchSem.acquire()
    #print("ding!")
    asyncBatchBuffer, asyncBatchBufferCropped = buildBatch(files, batchSize, k * 2, isTrain)
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
    
    
    padding = genModel.requiredPadding if IMAGE_USE_ADVANCED_PADDING else 0
    trainStepFunction = tf.function(trainStep, input_signature=(tf.TensorSpec(shape=[batchSize,k*2,k*2,genModel.numTextures*3], dtype=tf.float32),tf.TensorSpec(shape=[batchSize,k+2*padding,k+2*padding,genModel.numTextures*3], dtype=tf.float32),))
    
    threading.Thread(target=asyncLoadBatch, args=(files,batchSize,k,True,)).start()
    
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
        threading.Thread(target=asyncLoadBatch, args=(files,batchSize,k,True,)).start()
        batchSem.release()
        trainStepFunction(baseImages, croppedImages)
        if i%imageEveryXBatches == 0:
            saveTestImage(i)
        if i%lossStatsEveryXBatches == 0:
            logLossValues(i)

def saveTileableTextures(k, crop=True, filesuffix="", customInput=None, blockChunkedCall=True):
    global genModel
    
    genModel.tileLatentSpace = True
    if customInput != None:
        genInput = customInput
        k = customInput.shape[1]
    else:
        genInput = loadTestImage(k)
        
    if k > 256 and not blockChunkedCall:
        genOutput = genModel.chunkedCall(genInput)
    else:
        genOutput = genModel(genInput)
    
    for i in range(genModel.numTextures):
        if crop:
            texture = PIL.Image.fromarray((genOutput[0,k:3*k,k:3*k,i*3:(i+1)*3].numpy() * 255.0).astype("uint8"))
        else:
            texture = PIL.Image.fromarray((genOutput[0,:,:,i*3:(i+1)*3].numpy() * 255.0).astype("uint8"))
        
        texture.save(("%soutput%d%s.png" % (currentProjectPath, i, (("_" + filesuffix) if filesuffix else ""))))
    
    genModel.tileLatentSpace = False

#@tf.function
def createTileableTexturesFromInput(genInput, crop=True, blockChunkedCall=True):
    global genModel
    
    genModel.tileLatentSpace = True
    k = genInput.shape[1]
        
    if k > 256 and not blockChunkedCall:
        genOutput = genModel.chunkedCall(genInput)
    else:
        genOutput = genModel(genInput)

    genModel.tileLatentSpace = False    
    
    if crop:
        return genOutput[0,k:3*k,k:3*k,:]
    else:
        return genOutput[0,:,:,:]
    

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
    
    print("loading weights from %s" % (path))
    
    genModel.encoder.load_weights(path+"gen/enc")
    for i, resBlock in enumerate(genModel.residualBlocks):
        print("Loading resblock %d" % i)
        resBlock.load_weights(path+"gen/res"+str(i))
    for i, decoder in enumerate(genModel.decoders):
        print("Loading decoder %d" %i)
        decoder.load_weights(path+("gen/dec%d" % i))
    
    discModel.model.load_weights(path+"disc/weights")

def createModels():
    global genModel
    global discModel
    global baseImage
    global testImage
    
    baseImage = None
    testImage = None
    genModel = None
    
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
    loadConfiguration()
    createModels()
    loadWeights(currentProjectPath)
    
def saveModels():
    global genModel
    global discModel
    saveConfiguration()
    saveGeneratorWeights(currentProjectPath)
    saveDiscriminatorWeights(currentProjectPath)
    
def saveConfiguration():
    configDict = {
        "USE_L1" : USE_L1,
        "USE_LADV" : USE_LADV,
        "USE_LSTYLE" : USE_LSTYLE,

        "LAMBDA_L1" : LAMBDA_L1,
        "LAMBDA_LADV" : LAMBDA_LADV,
        "LAMBDA_LSTYLE" : LAMBDA_LSTYLE,
        
        "LSTYLE_LAYERS" : LSTYLE_LAYERS,
        "LADV_LAYERS" : LADV_LAYERS,

        "USE_PATCH_L1" : USE_PATCH_L1,

        "SILENT" : SILENT,

        "NUM_RESIDUAL_BLOCKS" : NUM_RESIDUAL_BLOCKS,
        "KERNEL_SIZE_RESIDUAL_BLOCKS" : KERNEL_SIZE_RESIDUAL_BLOCKS,
        "IMAGE_WIDTH_TRAINING" : IMAGE_WIDTH_TRAINING,
        "IMAGE_WIDTH_TESTING" : IMAGE_WIDTH_TESTING,
        "DISC_LEARNING_FACTOR" : DISC_LEARNING_FACTOR,
        "NUM_DECODER_CHUNKS" : NUM_DECODER_CHUNKS,
        "ENCODER_DEPTH" : ENCODER_DEPTH,
        "batchSize" : batchSize,
        
        "FEATUREMAP_INCREASE_FAC" : FEATUREMAP_INCREASE_FAC,
        "FEATUREMAP_MAX_SIZE" : FEATUREMAP_MAX_SIZE,
        "FEATUREMAP_START_SIZE" : FEATUREMAP_START_SIZE,
        
        "IMAGE_USE_ADVANCED_PADDING" : IMAGE_USE_ADVANCED_PADDING,
        "FORCE_INPUT_RESOLUTION" : FORCE_INPUT_RESOLUTION,
        "USE_UPSCALER" : USE_UPSCALER

        }
    print(configDict)
    with open(currentProjectPath + "project.conf", "w") as configFile:
        configFile.write(json.dumps(configDict, indent=4))
    
def loadConfiguration():
    global USE_L1
    global USE_LADV
    global USE_LSTYLE
    global LAMBDA_L1
    global LAMBDA_LADV
    global LAMBDA_LSTYLE
    global LSTYLE_LAYERS
    global LADV_LAYERS
    global USE_PATCH_L1
    global SILENT
    global NUM_RESIDUAL_BLOCKS
    global NUM_DECODER_CHUNKS
    global ENCODER_DEPTH
    global IMAGE_WIDTH_TESTING
    global IMAGE_WIDTH_TRAINING
    global DISC_LEARNING_FACTOR
    global batchSize
    global FEATUREMAP_INCREASE_FAC
    global FEATUREMAP_MAX_SIZE
    global FEATUREMAP_START_SIZE
    global IMAGE_USE_ADVANCED_PADDING
    global KERNEL_SIZE_RESIDUAL_BLOCKS
    global FORCE_INPUT_RESOLUTION
    global USE_UPSCALER
    
    configDict = {}
    try:
        with open(currentProjectPath + "project.conf") as configFile:
            configDict = json.loads(configFile.read())
    except:
        print(traceback.format_exc())
        print("Could not load config from file, reverting to defaults -> %s" % (currentProjectPath+currentProjectFolder))
        defaultConfiguration()


    USE_L1 = True if "USE_L1" not in configDict else configDict["USE_L1"]
    USE_LADV = True if "USE_LADV" not in configDict else configDict["USE_LADV"]
    USE_LSTYLE = True if "USE_LSTYLE" not in configDict else configDict["USE_LSTYLE"]

    LAMBDA_L1 = 10.0 if "LAMBDA_L1" not in configDict else configDict["LAMBDA_L1"]
    LAMBDA_LADV = 1.0 if "LAMBDA_LADV" not in configDict else configDict["LAMBDA_LADV"]
    LAMBDA_LSTYLE = 1.0 if "LAMBDA_LSTYLE" not in configDict else configDict["LAMBDA_LSTYLE"]
    
    LSTYLE_LAYERS = 1 if "LSTYLE_LAYERS" not in configDict else configDict["LSTYLE_LAYERS"]
    LADV_LAYERS = 1 if "LADV_LAYERS" not in configDict else configDict["LADV_LAYERS"]

    USE_PATCH_L1 = False if "USE_PATCH_L1" not in configDict else configDict["USE_PATCH_L1"]

    SILENT = False if "SILENT" not in configDict else configDict["SILENT"]

    NUM_RESIDUAL_BLOCKS = 5 if "NUM_RESIDUAL_BLOCKS" not in configDict else configDict["NUM_RESIDUAL_BLOCKS"]
    KERNEL_SIZE_RESIDUAL_BLOCKS = 3 if "KERNEL_SIZE_RESIDUAL_BLOCKS" not in configDict else configDict["KERNEL_SIZE_RESIDUAL_BLOCKS"]
    IMAGE_WIDTH_TRAINING = 128 if "IMAGE_WIDTH_TRAINING" not in configDict else configDict["IMAGE_WIDTH_TRAINING"]
    IMAGE_WIDTH_TESTING = 256 if "IMAGE_WIDTH_TESTING" not in configDict else configDict["IMAGE_WIDTH_TESTING"]
    DISC_LEARNING_FACTOR = 1.0 if "DISC_LEARNING_FACTOR" not in configDict else configDict["DISC_LEARNING_FACTOR"]
    NUM_DECODER_CHUNKS = 8 if "NUM_DECODER_CHUNKS" not in configDict else configDict["NUM_DECODER_CHUNKS"]
    ENCODER_DEPTH = 2 if "ENCODER_DEPTH" not in configDict else configDict["ENCODER_DEPTH"]
    batchSize = 1 if "batchSize" not in configDict else configDict["batchSize"]
    
    FEATUREMAP_INCREASE_FAC = 2 if "FEATUREMAP_INCREASE_FAC" not in configDict else configDict["FEATUREMAP_INCREASE_FAC"]
    FEATUREMAP_MAX_SIZE = 8192 if "FEATUREMAP_MAX_SIZE" not in configDict else configDict["FEATUREMAP_MAX_SIZE"]
    FEATUREMAP_START_SIZE = 64 if "FEATUREMAP_START_SIZE" not in configDict else configDict["FEATUREMAP_START_SIZE"]
    
    IMAGE_USE_ADVANCED_PADDING = True if "IMAGE_USE_ADVANCED_PADDING" not in configDict else configDict["IMAGE_USE_ADVANCED_PADDING"]
    FORCE_INPUT_RESOLUTION = False if "FORCE_INPUT_RESOLUTION" not in configDict else configDict["FORCE_INPUT_RESOLUTION"]
    USE_UPSCALER = False if "USE_UPSCALER" not in configDict else configDict["USE_UPSCALER"]
    
    print(configDict)
    
def defaultConfiguration():
    global USE_L1
    global USE_LADV
    global USE_LSTYLE
    global LAMBDA_L1
    global LAMBDA_LADV
    global LAMBDA_LSTYLE
    global LSTYLE_LAYERS
    global LADV_LAYERS
    global USE_PATCH_L1
    global SILENT
    global NUM_RESIDUAL_BLOCKS
    global NUM_DECODER_CHUNKS
    global ENCODER_DEPTH
    global IMAGE_WIDTH_TESTING
    global IMAGE_WIDTH_TRAINING
    global DISC_LEARNING_FACTOR
    global batchSize
    global FEATUREMAP_INCREASE_FAC
    global FEATUREMAP_MAX_SIZE
    global FEATUREMAP_START_SIZE
    global IMAGE_USE_ADVANCED_PADDING
    global KERNEL_SIZE_RESIDUAL_BLOCKS
    global FORCE_INPUT_RESOLUTION
    global USE_UPSCALER
    
    USE_L1 = True
    USE_LADV = True
    USE_LSTYLE = True

    LAMBDA_L1 = 10.0
    LAMBDA_LADV = 1.0
    LAMBDA_LSTYLE = 1.0
    
    LADV_LAYERS = 1
    LSTYLE_LAYERS = 1

    USE_PATCH_L1 = False

    SILENT = False

    NUM_RESIDUAL_BLOCKS = 5
    KERNEL_SIZE_RESIDUAL_BLOCKS = 3
    IMAGE_WIDTH_TRAINING = 128
    IMAGE_WIDTH_TESTING = 256
    DISC_LEARNING_FACTOR = 1.0
    NUM_DECODER_CHUNKS = 8
    ENCODER_DEPTH = 2
    batchSize = 1
    FEATUREMAP_START_SIZE = 64
    FEATUREMAP_INCREASE_FAC = 2
    FEATUREMAP_MAX_SIZE = 8192
    
    IMAGE_USE_ADVANCED_PADDING = True
    FORCE_INPUT_RESOLUTION = False
    USE_UPSCALER = False

    
def optimizedConfiguration():
    global USE_L1
    global USE_LADV
    global USE_LSTYLE
    global LAMBDA_L1
    global LAMBDA_LADV
    global LAMBDA_LSTYLE
    global LSTYLE_LAYERS
    global LADV_LAYERS
    global USE_PATCH_L1
    global SILENT
    global NUM_RESIDUAL_BLOCKS
    global NUM_DECODER_CHUNKS
    global ENCODER_DEPTH
    global IMAGE_WIDTH_TESTING
    global IMAGE_WIDTH_TRAINING
    global DISC_LEARNING_FACTOR
    global batchSize
    global FEATUREMAP_INCREASE_FAC
    global FEATUREMAP_MAX_SIZE
    global FEATUREMAP_START_SIZE
    global IMAGE_USE_ADVANCED_PADDING
    global KERNEL_SIZE_RESIDUAL_BLOCKS
    global FORCE_INPUT_RESOLUTION
    global USE_UPSCALER
    
    USE_L1 = True
    USE_LADV = True
    USE_LSTYLE = True

    LAMBDA_L1 = 1.0
    LAMBDA_LADV = 1.0
    LAMBDA_LSTYLE = 10.0
    
    LADV_LAYERS = 1
    LSTYLE_LAYERS = 2

    USE_PATCH_L1 = False

    SILENT = False

    NUM_RESIDUAL_BLOCKS = 6
    KERNEL_SIZE_RESIDUAL_BLOCKS = 7
    IMAGE_WIDTH_TRAINING = 128
    IMAGE_WIDTH_TESTING = 256
    DISC_LEARNING_FACTOR = 0.5
    NUM_DECODER_CHUNKS = 8
    ENCODER_DEPTH = 2
    batchSize = 1
    FEATUREMAP_START_SIZE = 32
    FEATUREMAP_INCREASE_FAC = 2
    FEATUREMAP_MAX_SIZE = 8192
    
    IMAGE_USE_ADVANCED_PADDING = True
    FORCE_INPUT_RESOLUTION = True
    USE_UPSCALER = True

def setProject(projectName):
    global baseImage
    global testImage
    global currentProjectPath
    baseImage = None
    testImage = None
    oldProjectPath = currentProjectPath
    currentProjectPath = currentProjectFolder+projectName+"/"
    try:
        os.mkdir(currentProjectPath)
    except:
        pass
    try:
        os.mkdir(currentProjectPath+"images")
        filesToCopy = os.listdir(oldProjectPath+"images/")
        for file in filesToCopy:
            shutil.copyfile(oldProjectPath+"images/"+file, currentProjectPath+"images/"+file)
    except:
        pass

def loadImageStack():
    global baseImage
    global upscalerImages
    
    baseImage = None
    upscalerImages = None
    
    size = (-1,-1)
    
    for imgFilename in os.listdir(currentProjectPath+"images/"):
        if os.path.isfile(currentProjectPath+"images/"+imgFilename):
            try:
                image = PIL.Image.open(currentProjectPath+"images/"+imgFilename)
                upscalerImage = None
                
                #rescale to 512px if the configuration says so
                try:
                    if FORCE_INPUT_RESOLUTION:
                        imageSize = image.size
                        if size == (-1,-1):
                            size = imageSize
                        elif imageSize != size:
                            print("WARNING: input images are of different sizes! input images MUST be exactly the same size for this program to work reliably")
                        newSizeFactor = 512.0 / min(imageSize)
                        newSize = tuple(int(newSizeFactor * s) for s in imageSize)
                        if USE_UPSCALER:
                            upscalerImage = image.resize(tuple(int(s*2) for s in newSize), PIL.Image.LANCZOS)
                        image = image.resize(newSize, PIL.Image.LANCZOS)
                    
                    
                    #convert to tensor (?) TODO: check if there is a cleaner option
                    image = tf.image.crop_to_bounding_box(image, 0,0, image.getbbox()[3], image.getbbox()[2])
                    #remove alpha channel
                    image = image[:,:,:3]
                    #add to stack
                    baseImage = image if baseImage == None else tf.concat([baseImage,image], axis=2)
                    
                    if USE_UPSCALER:
                        upscalerImage = tf.image.crop_to_bounding_box(upscalerImage, 0,0, upscalerImage.getbbox()[3], upscalerImage.getbbox()[2])
                        upscalerImage = upscalerImage[:,:,:3]
                        upscalerImages = upscalerImage if upscalerImages == None else tf.concat([upscalerImages,upscalerImage], axis=2)
                except:
                    print("Error while trying to resize and convert the input image %s", imgFilename)
            except:
                print("Warning: could not load base image with name %s, skipping..." % imgFilename)
    
    
def saveImage(inputData, filename):
    PIL.Image.fromarray((inputData.numpy() * 255.0).astype("uint8")).save(currentProjectPath+filename+".png")
    
def stdLearning(saveFinalTexture=False):
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
    train(40000,50001,0.000008,IMAGE_WIDTH_TRAINING,500)
    saveModels()
    tf.keras.backend.clear_session()
    if saveFinalTexture:
        saveTileableTextures(1024)#TODO: use maximum input texture size instead of hard-coded value
    print("finished training in %d minutes. Bye!" % int((time.time()-sTime)/60))
    
    
def optimizedLearning():
    sTime = time.time()
    DISC_LEARNING_FACTOR = 0.5
    FEATUREMAP_START_SIZE = 32
    LAMBDA_L1 = 1.0
    LAMBDA_LADV = 1.0
    LAMBDA_LSTYLE = 20.0
    NUM_RESIDUAL_BLOCKS = 6
    LSTYLE_LAYERS = 3
    KERNEL_SIZE_RESIDUAL_BLOCKS = 7
    IMAGE_USE_ADVANCED_PADDING = True
    
    tf.keras.backend.clear_session()
    createModels()
    
    train(0,2000,0.001,192,500,100)
    LAMBDA_LSTYLE = 10.0
    LSTYLE_LAYERS = 2
    
    train(2000,10000,0.001,128,500,100)
    train(10000,17000,0.0002,128,500,100)
    train(17000,22000,0.00005,128,500,100)
    saveModels()
    train(22000,25000,0.00002,128,500,100)
    train(25000,26001,0.00001,128,500,100)
    saveModels()
    
    plotLosses(10,True)
    print("finished optimized training in %d minutes. Bye!" % int((time.time()-sTime)/60))

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
    try:
        os.mkdir(currentProjectPath)
    except:
        pass
    inputNoise = tf.random.uniform(shape=(1,startK,startK,genModel.numTextures*3), minval=0.0, maxval=1.0)
    saveImage(inputNoise[0,:,:,:3], ("n%d" % startK))
    output = None
    iterK = startK
    for i in range(iterations):
        if i == 0:
            output = genModel(inputNoise)
        else:
#             if iterK > 200:
#                 output = genModel.chunkedCall(output)
#             else:
#                 output = genModel(output)
            output = genModel(output)
        iterK *= 2
        saveImage(output[0,:,:,:3], ("n%d" % iterK))
    if makeFinalStack:
        saveTileableTextures(0, True, "_fromnoise", output, True)
    currentProjectPath = originalPath
    
def noiseSweep(k=256, percentageStride=1):
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
    
    for i in range(0,101,percentageStride):
        infoFactor = 1 - (i/100)
        noiseFactor = i/100
        mix = infoFactor * info + noiseFactor * noise
        saveImage(mix[0,:,:,:3], ("mix%d" % i))
        saveTileableTextures(0, True, ("noise%dpercent" %i), mix, True)
    currentProjectPath = originalPath
    
def lambdaStudy(basename, startAtPermutation=0):
    global LAMBDA_L1
    global LAMBDA_LADV
    global LAMBDA_STYLE
    
    sTime = time.time()
    
    permutation = -1
    
    for ilstyle in [1,10,50]:
        for iladv in [1,10,50]:
            for il1 in [1,10,50]:
                permutation += 1
                if permutation < startAtPermutation:
                    continue #this is added to make it possible to resume the study on our cluster which automatically closes the session after 4 hours
                LAMBDA_LADV = float(iladv)
                LAMBDA_L1 = float(il1)
                LAMBDA_STYLE = float(ilstyle)
                setProject("%s_l1-%d_ladv-%d_lstyle-%d" % (basename, il1, iladv, ilstyle))
                createModels()
                clearLossLog()
                tf.keras.backend.clear_session()
                train(0,10000,0.0002,IMAGE_WIDTH_TRAINING,500)
                saveModels()
                tf.keras.backend.clear_session()
    
    print("finished lambda study in %d minutes. Bye!" % int((time.time()-sTime)/60))
                
def inputTextureSizeStudy(projectNameList):
    for projectName in projectNameList:
        setProject(projectName)
        createModels()
        clearLossLog()
        tf.keras.backend.clear_session()
        train(0,15000,0.0002,IMAGE_WIDTH_TRAINING,500)
        saveModels()
        train(15000,30001,0.0002,IMAGE_WIDTH_TRAINING,500)
        saveModels()
        tf.keras.backend.clear_session()
    

currentProjectFolder = "projects/"
currentProjectPath = currentProjectFolder + "default/"
upscalerImages = None
loadImageStack()
files = os.listdir(currentProjectPath+"images")
setProject("default")
testImage = loadTestImage(256)
print("initialized!")




# import upscaler as up
# setProject("rock512")
# loadModels()
# up.createUpscaleModel(baseImage.shape[2])
# smallImage = tf.expand_dims(tf.image.resize(baseImage, (baseImage.shape[0]//2, baseImage.shape[1]//2)) / 255., axis=0)
# print(smallImage)
# 
# up.trainOnImage(baseImage, 10000, 0.001)
# upscaledImage = up.upscaleModel(smallImage)
# upscaledImage = tf.squeeze(upscaledImage)
# saveImage(upscaledImage[:,:,:3],"upscale01")
# up.trainOnImage(baseImage, 10000, 0.001)
# upscaledImage = up.upscaleModel(smallImage)
# upscaledImage = tf.squeeze(upscaledImage)
# saveImage(upscaledImage[:,:,:3],"upscale02")
# up.trainOnImage(baseImage, 10000, 0.001)
# upscaledImage = up.upscaleModel(smallImage)
# upscaledImage = tf.squeeze(upscaledImage)
# saveImage(upscaledImage[:,:,:3],"upscale03")
# up.trainOnImage(baseImage, 10000, 0.0001)
# upscaledImage = up.upscaleModel(smallImage)
# upscaledImage = tf.squeeze(upscaledImage)
# saveImage(upscaledImage[:,:,:3],"upscale04")
# up.trainOnImage(baseImage, 10000, 0.0001)
# upscaledImage = up.upscaleModel(smallImage)
# upscaledImage = tf.squeeze(upscaledImage)
# saveImage(upscaledImage[:,:,:3],"upscale05")
# up.trainOnImage(baseImage, 10000, 0.0001)
# upscaledImage = up.upscaleModel(smallImage)
# upscaledImage = tf.squeeze(upscaledImage)
# saveImage(upscaledImage[:,:,:3],"upscale06")
