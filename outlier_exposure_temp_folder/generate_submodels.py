import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.models import load_model

root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma'
orig = load_model(root+'/model/model_mnist.h5')

layer_list = orig.layers
non_trainable = ['dropout', 'activation']
for l, ltype in enumerate(layer_list):
    layers = layer_list[0:l+1]
    n_model = Sequential()
    for layer in layers:
        if 'conv2d' in layer.name:
            n_model.add(layer)
            n_model.add(Activation(activation='relu'))
        else:
            n_model.add(layer)
    n_model.add(Flatten())

    for layer in n_model.layers:
        layer.trainable = False

    n_model.add(Dense(10, activation='softmax'))
    print(n_model.summary())
    n_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics='accuracy')
    #n_model.save("submodel_{}_mnist.h5".format(l))

