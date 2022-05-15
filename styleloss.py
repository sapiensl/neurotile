import tensorflow as tf
from tensorflow.keras import layers


def createVGGLayers(layerNames):
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layerNames]
    model = tf.keras.Model([vgg.input],outputs)
    return model

def calculateGramMatrix(inputTensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', inputTensor, inputTensor)
    inputShape = tf.shape(inputTensor)
    numLocations=tf.cast(inputShape[1]*inputShape[2], tf.float32)
    return result/(numLocations)

def calculateStyleLoss(model, targetInput, lossyInput):
    global testI
    targetOutputs = model(targetInput)
    lossyOutputs = model(lossyInput)
    targetOutputsGram = [calculateGramMatrix(i) for i in targetOutputs]
    lossyOutputsGram = [calculateGramMatrix(i) for i in lossyOutputs]
    #this verision uses the latent image size of the respective layer as a quotient
    #styleLoss = tf.add_n([tf.reduce_sum(tf.reduce_sum((lOG - tOG)**2)) / ((2 * lO.shape[1] * lOG.shape[1])**2) * (1000.0 / (lOG.shape[1]**2)) for lOG, tOG, lO in zip(lossyOutputsGram, targetOutputsGram, lossyOutputs)])
    
    #this version uses the size of the generated image as a quotient. Results are much better this way
    styleLoss = tf.add_n([tf.reduce_sum(tf.reduce_sum((lOG - tOG)**2)) / ((2 * lossyInput.shape[1] * lOG.shape[1])**2) * (1000.0 / (lOG.shape[1]**2)) for lOG, tOG, lO in zip(lossyOutputsGram, targetOutputsGram, lossyOutputs)])
    return styleLoss
    

def createStyleModel():
    return StyleModel()

class StyleModel(tf.keras.models.Model):
    def __init__(self):
        super(StyleModel,self).__init__()
        self.vgglayers = createVGGLayers(['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1','block5_conv1'])
        self.vgglayers.trainable = False
        #self.conversionLayer = layers.Activation("linear", dtype="float32")
        
    def call(self, inputs):
        inputs = inputs*255.0
        inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgglayers(inputs)
        #outputs = self.conversionLayer(outputs)
        #outputs = [calculateGramMatrix(layer) for layer in outputs]
        return outputs
