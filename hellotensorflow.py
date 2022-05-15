import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist


(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
print(xTrain.shape())

print("Hello World")