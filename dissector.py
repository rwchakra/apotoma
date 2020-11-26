"""Stages:

    0. Function to generate ground truths:
        Input: model, x_test, test_pred
        Output: 1-D binary array, 1 if classified correctly, zero if classified incorrectly

    1. Function to create sub-models
    Input: model (trained), list of layer numbers, no. of output nodes (num_classes)
    Output: list of sub_models

    2. Function to train sub models (Only final layer connections to fully connected). Freeze other weights.
        Input: list of sub_models, x_train
        Output: list of trained sub_models

        Number of epochs? Unclear, paper does not specify.

    3. Function to calculate prediction profile: One softmax vector per sub-model
        Input: List of trained sub_models
        Outputs: k Array of softmax arrays, one for each sub_model, shape: #inputs * #classes * #sub_models

    4. Function to calculate SV score
        Input: a) k array of softmax arrays, one for each sub_model, shape: #inputs * #classes * #sub_models
               b) test_pred: Array of test predictions of original model M

        Output: Array of SV scores of length k, hopefully ordered by shallow to deep sub models.

    5. Function to calculate PV score for all x_test
        Input: a) x_test, array_sv_score, weights_array
        Output: Array of PV scores in (0,1).

    6. Function to calculate weights array [3 options: Logarithmic, Linear, Exponential]:
        Input: a) Type of weight growth
               b) Number k of specific layer

        Output: array of weights, shape: (#sub_models,)

    7. Function calculate AUC score:
        Inputs: a) generate_ground_truth(model, test_pred, x_test)
                b) PV score array
        Output: roc_auc_score(labels, pv_score_array)
    """

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Dense, Flatten
import os
from tqdm import tqdm

class Dissector():

    def __init__(self, model: tf.keras.Model, model_path: str, sub_model_path: str):

        self.model = model
        self.model_path = model_path
        self.sub_model_path = sub_model_path


    def generate_ground_truth(self, x_test: tf.data.Dataset, y_test: tf.data.Dataset):

        #model = load_model(self.args['model_path'])
        test_preds = self.model.predict_classes(x_test)
        labels = np.argmax(y_test, axis=1)
        corr = np.where(test_preds == labels)
        incorr = np.where(test_preds != labels)
        labels[corr] = 1
        labels[incorr] = 0

        return test_preds, labels

    def generate_sub_models(self, layer_list: []):

        for _, l in enumerate(layer_list):
            n_model = Sequential()
            for layer in self.model.layers[0:l]:
                n_model.add(layer)

            n_model.add(Flatten())
            for layer in n_model.layers:
                layer.trainable = False

            n_model.add(Dense(10, activation='softmax'))
            n_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            n_model.save(self.sub_model_path+"submodel_{}_lenet4_{}.h5".format(l, 'mnist'))

    def train_sub_models(self, x_train: tf.data.Dataset, y_train: tf.data.Dataset, epochs: int):


        sub_models = os.listdir(self.sub_model_path)

        for i, m in enumerate(tqdm(sub_models)):
            s_model = load_model(self.sub_model_path+m)
            s_model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                verbose=0,
                validation_split=0.2,
            )

            s_model.save(self.sub_model_path+m)


    def sv_score(self, x_test: tf.data.Dataset):

        m = load_model(self.model_path)
        test_preds = m.predict_classes(x_test)
        sub_models = os.listdir(self.sub_model_path)
        scores = {}

        for i, m in enumerate(sub_models):
            sv_scores = []
            s_model = load_model(self.sub_model_path + m)

            activations = s_model.predict(x_test)
            preds = s_model.predict_classes(x_test)

            for i, p in enumerate(preds):

                if np.argmax(activations[i]) == test_preds[i]:

                    i_x = np.max(activations[i])
                    i_sh = np.sort(activations[i])[::-1][1]

                    sv_score = i_x / (i_x + i_sh)

                else:
                    i_h = np.max(activations[i])
                    i_x = activations[i][test_preds[i]]

                    sv_score = 1 - i_h / (i_h + i_x)

                sv_scores.append(sv_score)

            scores[m] = sv_scores

        return scores

    def get_weights(self, growth_type: str, alpha: float):

        sub_models = os.listdir(self.sub_model_path)
        weights = {}

        assert growth_type in ['linear', 'logarithmic', 'exponential'], "Invalid weight growth type"
        for m in sub_models:
            l_number = int(m.split("_")[1]) + 1

            if growth_type == 'linear':

                y = alpha * l_number + 1

            elif growth_type == 'logarithmic':

                y = alpha * np.log(l_number) + 1

            else:

                y = np.exp(alpha * l_number)

            weights[m] = y

        return weights

    def pv_scores(self, weights: {}, scores: {}):

        pv_scores = np.zeros(shape=(10000,))
        for sub_model, sv in scores.items():
            weighted_score = weights[sub_model] * np.array(sv)
            pv_scores += weighted_score

        pv_scores = pv_scores / sum(weights.values())

        return pv_scores

