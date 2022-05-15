import os
import random
import PIL
import PIL.Image
import tensorflow as tf

def buildBatch(files, batchSize, maxImageWidth):
    images = []
    randomFiles = random.choices(files, k=batchSize)
    for file in randomFiles:
        img = PIL.Image.open("images/"+file)
        cropped = tf.image.crop_to_bounding_box(img, 0,0,100,100)
        images.append(cropped)
    return images