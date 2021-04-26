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
model = load_model(root+'/model/model/mnist_models/model_mnist_10.h5')

print(model.summary())
l_names = ['conv2d_16', 'conv2d_17', 'max_pooling2d_8', 'dense_16'] #Pass in all layers except final dense softmax

for i, l_name in enumerate(l_names):
            mid = model.get_layer(l_name).output
            if ('conv2d' in l_name or 'dense' in l_name): #Add activation for trainable layers
                act = Activation('relu')(mid)
            else:
                act = mid
            if 'dense' not in l_name: #Flatten only if not already a dense layer
                flat = Flatten()(act)
            else:
                flat = mid
            final = Dense(10, activation='softmax')(flat)
            n_model = tf.keras.Model(model.input, final)
            for l in n_model.layers[:-1]:
                l.trainable = False

            print(n_model.summary())
            n_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics='accuracy')
            n_model.save(root+"/submodels_dissector_latest/model_mnist/model_10/submodel_{}.h5".format(i))

# while i < len(layer_list):
#     layers = layer_list[0:i+1]
#     n_model = Sequential()
#     for layer in layers:
#
#         if (('activation' in layer.name) or ('dropout' in layer.name)):
#             continue
#         elif 'conv2d' in layer.name:
#             n_model.add(layer)
#             n_model.add(Activation(activation='relu'))
#         else:
#             n_model.add(layer)
#
#     if ('flatten' not in layer.name) and ('dense' not in layer.name):
#         n_model.add(Flatten())
#
#     for layer in n_model.layers:
#         layer.trainable = False
#
#     if layer.get_config() != layer_list[-1].get_config():
#         n_model.add(Dense(10, activation='softmax'))
#     print(n_model.summary())
#     n_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics='accuracy')
#     i += 2
#     #n_model.save("submodel_{}_mnist.h5".format(l))
