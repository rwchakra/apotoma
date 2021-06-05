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


    5. Function to calculate weights array [3 options: Logarithmic, Linear, Exponential]:
        Input: a) Type of weight growth
               b) Number k of specific layer

        Output: array of weights, shape: (#sub_models,)

    6. Function to calculate PV score for all x_test
        Input: a) x_test, array_sv_score, weights_array
        Output: Array of PV scores in (0,1).

    7. Function calculate AUC score:
        Inputs: a) generate_ground_truth(model, test_pred, x_test)
                b) PV score array
        Output: roc_auc_score(labels, pv_score_array)
    """

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tqdm import tqdm
import argparse
import re


class Dissector:

    def __init__(self, model: tf.keras.Model, config: argparse.Namespace):

        self.model = model
        self.config = config

    def generate_ground_truth(self, x_test: tf.data.Dataset, y_test: tf.data.Dataset):

        # g_truths = np.ones(shape=x_test.shape[0], dtype = bool)
        # model = load_model(self.args['model_path'])
        test_preds = self.model.predict(x_test)
        test_preds = np.argmax(test_preds, axis=1)
        labels = np.argmax(y_test, axis=1)
        corr = np.where(test_preds == labels)
        incorr = np.where(test_preds != labels)
        labels[corr] = 1
        labels[incorr] = 0

        return test_preds, labels

    def generate_sub_models(self, layer_list: []):

        """Two options below: Either a sequential (for list of ints) or a functional (list of strs)"""

        # for i, l_name in enumerate(l_names):
        #     mid = self.model.get_layer(l_name).output
        #     flat = Flatten()(mid)
        #     final = Dense(10, activation='softmax')(flat)
        #     n_model = Model(self.model.input, final)
        #     for l in n_model.layers[:-1]:
        #         l.trainable = False
        #
        #     n_model.compile(loss=self.config.loss, optimizer=self.config.optimizer, metrics=self.config.metrics)
        #     n_model.save(self.config.sub_model_path + "submodel_"+self.config.model_type+"_{}.h5".format(l, self.config.dataset_name))

        for _, l in enumerate(layer_list):
            n_model = Sequential()
            for layer in self.model.layers[0:l]:
                n_model.add(layer)

            n_model.add(Flatten())
            for layer in n_model.layers:
                layer.trainable = False

            n_model.add(Dense(self.config.num_classes, activation='softmax'))
            n_model.compile(loss=self.config.loss, optimizer=self.config.optimizer, metrics=self.config.metrics)
            n_model.save(self.config.sub_model_path + "submodel_"+self.config.model_type+"_{}.h5".format(l, self.config.dataset_name))

    def train_sub_models(self, x_train: tf.data.Dataset, y_train: tf.data.Dataset, epochs: int):

        sub_models = os.listdir(self.config.sub_model_path)

        for i, m in enumerate(tqdm(sub_models)):
            s_model = load_model(self.config.sub_model_path + m)
            s_model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                verbose=0,
                validation_split=self.config.val_split,
            )

            s_model.save(self.config.sub_model_path + m)

    def sv_score(self, x_test: tf.data.Dataset):
        """Check out 'Prediction confidence score (PCS)' - michael will give you reference for msc thesis"""

        m = load_model(self.config.model_path)
        m_preds = m.predict(x_test)
        #Uncomment below if running for CIFAR. Not required for MNIST

        f = m_preds
        #f = np.exp(m_preds - np.amax(m_preds, axis=1)[:, None])
        #f = f / np.sum(f[:, None], axis=-1)

        test_preds = np.argmax(f, axis=1)
        sub_models = os.listdir(self.config.sub_model_path)
        print(sub_models)
        scores = np.empty(shape=(len(sub_models), (x_test.shape[0])))

        for index, m in enumerate(sub_models):
            sv_scores = []
            s_model = load_model(self.config.sub_model_path + m)

            activations = s_model.predict(x_test)
            preds = np.argmax(activations, axis=1)

            for i, p in enumerate(preds):

                if np.argmax(activations[i]) == test_preds[i]:
                    sv_score = self._calc_sv_for_match(activations, i)
                else:
                    sv_score = self._calc_sv_for_non_match(activations, i, test_preds)

                sv_scores.append(sv_score)

            scores[index] = np.array(sv_scores)

        return scores

    @staticmethod

    def _calc_sv_for_non_match(activations, i, test_preds):
        i_h = np.max(activations[i])
        i_x = activations[i][test_preds[i]]
        sv_score = 1 - i_h / (i_h + i_x)
        return sv_score

    @staticmethod

    def _calc_sv_for_match(activations, i):
        i_x = np.max(activations[i])
        i_sh = np.sort(activations[i])[::-1][1]
        sv_score = i_x / (i_x + i_sh)
        return sv_score

    def get_weights(self, growth_type: str, alpha: float):

        sub_models = os.listdir(self.config.sub_model_path)
        print(sub_models)
        weights = np.empty(shape=(len(sub_models),))

        assert growth_type in ['linear', 'logarithmic', 'exponential'], "Invalid weight growth type"
        for i, m in enumerate(sub_models):
            l_number = int(int(re.search(r'\d+', m).group())) + 1

            if growth_type == 'linear':

                #y = alpha * l_number + 1
                y = l_number
            elif growth_type == 'logarithmic':

                #y = alpha * np.log(l_number) + 1
                y = np.log(l_number)

            else:

                #y = np.exp(alpha * l_number)
                y = np.exp(l_number)

            weights[i] = y

        return weights


    def pv_scores(self, weights: np.ndarray, scores: np.ndarray):

        pv_scores = np.sum(np.multiply(scores, weights[:, np.newaxis]), axis=0) / np.sum(weights)

        return pv_scores
