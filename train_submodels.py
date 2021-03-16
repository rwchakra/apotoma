import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10
from tqdm import tqdm
from tensorflow.keras import utils
import argparse

root = "D:\Rwiddhi\Github"
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train = x_train.reshape(-1, 28, 28, 1)
#x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = (x_train / 255.0)
x_test = (x_test / 255.0)

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

sub_models = os.listdir("D:\Rwiddhi\Github\submodels_dissector\model_outexp_nosmcifar")[:-1]

for i, m in enumerate(sub_models):
    s_model = load_model("D:\Rwiddhi\Github\submodels_dissector\model_outexp_nosmcifar\\"+m)
    s_model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_split=0.2,
    )

    s_model.save("D:\Rwiddhi\Github\submodels_dissector\model_outexp_nosmcifar\\" + m)