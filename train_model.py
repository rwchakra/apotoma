import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback

CLIP_MIN = -0.5
CLIP_MAX = 0.5

"""class SaveWeights(Callback):

    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights("./weights/"+name)
        self.batch += 1"""

class MNISTModel():

    def __init__(self, args):

        self.args = args

    def train(self):
        if self.args.d == "mnist":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)

            layers = [
                Conv2D(64, (3, 3), padding="valid", input_shape=(28, 28, 1)),
                Activation("relu"),
                Conv2D(64, (3, 3)),
                Activation("relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.5),
                Flatten(),
                Dense(128),
                Activation("relu"),
                Dropout(0.5),
                Dense(10),
            ]

        elif self.args.d == "cifar":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            layers = [
                Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)),
                Activation("relu"),
                Conv2D(32, (3, 3), padding="same"),
                Activation("relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), padding="same"),
                Activation("relu"),
                Conv2D(64, (3, 3), padding="same"),
                Activation("relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), padding="same"),
                Activation("relu"),
                Conv2D(128, (3, 3), padding="same"),
                Activation("relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.5),
                Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                Activation("relu"),
                Dropout(0.5),
                Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                Activation("relu"),
                Dropout(0.5),
                Dense(10),
            ]

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        y_train = utils.to_categorical(y_train, 10)
        y_test = utils.to_categorical(y_test, 10)

        #Replace for loop
        model = Sequential()
        for layer in layers:
            model.add(layer)
        model.add(Activation("softmax"))

        print(model.summary())
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(
            x_train,
            y_train,
            epochs=20,
            batch_size=128,
            shuffle=True,
            verbose=1,
            validation_data=(x_test, y_test),
        )

        model.save("./model/model_{}.h5".format(args.d))



class LeNet4():

    def __init__(self, args):
        self.args = args

    def train(self):

        if self.args.d == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        y_train = utils.to_categorical(y_train, 10)
        y_test = utils.to_categorical(y_test, 10)

        x_train, x_test = ZeroPadding2D(padding=2)(x_train), ZeroPadding2D(padding=2)(x_test)

        print(x_train.shape[1:])



        model = Sequential()
        model.add(Conv2D(4, (5, 5), activation='relu', input_shape=x_train.shape[1:], name='conv_1'))
        model.add(AveragePooling2D())
        model.add(Conv2D(16, (5, 5), activation='relu', name='conv_2'))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(120, activation='relu', name='dense_1'))
        model.add(Dense(10, activation='softmax'))

        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=128,
            shuffle=True,
            verbose=1,
            validation_split=0.2,
        )

        model.save("./model/model_lenet4_{}.h5".format(args.d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"


    #model = MNISTModel(args)
    model = LeNet4(args)
    #model = MNISTModel(args)

    model.train()
