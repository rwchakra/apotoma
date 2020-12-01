from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from apotoma.novelty_score import NoveltyScore
from scipy.stats import gaussian_kde
from tqdm import tqdm
import os


class SurpriseAdequacy(NoveltyScore):

    def __init__(self, model: tf.keras.Model, train_data: tf.data.Dataset, args: {}) -> None:
        super().__init__(model, train_data, args)
        self.train_ats, self.train_pred, self.class_matrix = None, None, {}

    def _get_saved_path(self, ds_name):
        """Determine saved path of ats and pred

        Args:
            base_path (str): Base save path.
            dataset (str): Name of dataset.
            dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
            layer_names (list): List of layer names.

        Returns:
            ats_path: File path of ats.
            pred_path: File path of pred (independent of layers)
        """

        joined_layer_names = "_".join(self.args['layer_names'])

        return (
            os.path.join(
                self.args['saved_path'],
                self.args['d'] + "_" + ds_name + "_" + joined_layer_names + "_ats" + ".npy",
            ),
            os.path.join(self.args['saved_path'], self.args['d'] + "_" + ds_name + "_pred" + ".npy"),
        )

    # Returns ats and returns predictions
    def _load_or_calculate_ats(self, dataset: tf.data.Dataset, ds_name: str, use_cache: bool) -> Tuple[np.ndarray, np.ndarray]:

        """Determine activation traces train, target, and test datasets

                Args:
                    dataset (ndarray): x_train or x_test or x_target.
                    ds_name (str): Type of dataset: Train, Test, or Target.
                    use_cache (bool): Use stored files to load activation traces or not

                Returns:
                    ats (ndarray): Activation traces (Shape of num_examples * num_nodes).
                    pred (ndarray): 1-D Array of predictions

        """
        print(f"Calculating the ats for {ds_name} dataset")

        saved_target_path = self._get_saved_path(ds_name)
        if os.path.exists(saved_target_path[0]) and use_cache:
            return self._load_ats(ds_name, saved_target_path)
        else:
            ats, pred = self._calculate_ats(dataset, ds_name)

            if saved_target_path is not None:
                np.save(saved_target_path[0], ats)
                np.save(saved_target_path[1], pred)
                print("Cached the ["+ds_name+"]"+" ats and predictions")

            return ats, pred

    def _calculate_ats(self, dataset, ds_name):
        temp_model = Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(layer_name).output for layer_name in self.args['layer_names']]
        )

        if self.args['is_classification']:
            # p = Pool(num_proc)
            print("[" + ds_name + "]" + " Model serving")
            # Shape of len(ds_name_pred): predictions for the ds_name set
            pred = self.model.predict_classes(dataset, batch_size=self.args['batch_size'], verbose=1)
            if len(self.args['layer_names']) == 1:
                # layer_outputs is 60,000 * 10, since there are 10 nodes in activation_3
                layer_outputs = [
                    temp_model.predict(dataset, batch_size=self.args['batch_size'], verbose=1)
                ]

            else:
                layer_outputs = temp_model.predict(
                    dataset, batch_size=self.args['batch_size'], verbose=1
                )

            print("Processing " + ds_name + " ATs")
            ats = None
            for layer_name, layer_output in zip(self.args['layer_names'], layer_outputs):
                print("Layer: " + layer_name)
                if layer_output[0].ndim == 3:
                    # For convolutional layers
                    layer_matrix = np.array(
                        map(lambda x: [np.mean(x[..., j]) for j in range(x.shape[-1])],
                            [layer_output[i] for i in range(len(dataset))])
                    )
                else:
                    layer_matrix = np.array(layer_output)

                if ats is None:
                    # Initially ats is None, so ats is 60,000 * 10
                    ats = layer_matrix
                else:
                    ats = np.append(ats, layer_matrix, axis=1)
                    layer_matrix = None

        return ats, pred

    def _load_ats(self, ds_name, saved_target_path):
        print("Found saved {} ATs, skip serving".format(ds_name))
        # In case train_ats is stored in a disk
        ats = np.load(saved_target_path[0])
        pred = np.load(saved_target_path[1])
        return ats, pred

    def _load_or_calc_train_ats(self, use_cache = False):

        """Load or get actviation traces of training inputs

                        Args:
                            use_cache: To load stored files or not

                        Returns:
                            None. train_ats and train_pred are init() variables in super class NoveltyScore.

        """

        saved_train_path = self._get_saved_path("train")

        if os.path.exists(saved_train_path[0]) and use_cache:
            print("Found saved {} ATs, skip serving".format("train"))
            # In case train_ats is stored in a disk
            self.train_ats, self.train_pred = np.load(saved_train_path[0]), np.load(saved_train_path[1])

        else:
            self.train_ats, self.train_pred = self._load_or_calculate_ats(dataset=self.train_data, ds_name="train", use_cache=use_cache)

    def prep(self):

        """

        Prepare class matrix from training activation traces. Class matrix is a dictionary
        with keys as labels and values as lists of positions as predicted by model

                        Args:
                            None

                        Returns:
                            None. class_matrix is init() variable of NoveltyScore

        """
        self._load_or_calc_train_ats()
        for i, label in enumerate(self.train_pred):
            if label not in self.class_matrix:
                self.class_matrix[label] = []
            self.class_matrix[label].append(i)

    def clear_cache(self, saved_path: str):

        """

            Delete files of activation traces.

            Args:
                saved_path(str): Base directory path

        """

        files = [f for f in os.listdir(saved_path) if f.endswith('.npy')]
        for f in files:
            os.remove(os.path.join(saved_path, f))


class LSA(SurpriseAdequacy):

    def calc(self, target_data: tf.data.Dataset, ds_name: str, use_cache = False):

        """
        Return LSA values for target. Note that target_data here means both test and adversarial data. Separate calls in main.

                        Args:
                            target_data (ndarray): x_test or x_target.
                            ds_name (str): Type of dataset: Train, Test, or Target.
                            use_cache (bool): Use stored files to load activation traces or not

                        Returns:
                            lsa (float): The scalar LSA value

        """
        target_ats, target_pred = self._load_or_calculate_ats(dataset=target_data, ds_name=ds_name, use_cache = use_cache)

        kdes, removed_rows = self._calc_kdes()
        return self._calc_lsa(target_ats, target_pred, kdes, removed_rows, ds_name)

    def _calc_kdes(self) -> Tuple[List[object], List[int]]:

        """
            Determine Gaussian KDE for each label and list of removed rows based on variance threshold, if any.

            Args:
                target_data (ndarray): x_test or x_target.
                ds_name (str): Type of dataset: Train, Test, or Target.
                use_cache (bool): Use stored files to load activation traces or not

            Returns:
                kdes: Dict - labels are keys, values are scipy kde objects
                removed_rows: Array of positions of removed rows

        """

        removed_rows = []

        if self.args['is_classification']:

            for label in range(self.args['num_classes']):
                # Shape of (num_activation nodes x num_examples_by_label)
                row_vectors = np.transpose(self.train_ats[self.class_matrix[label]])
                positions = np.where(np.var(row_vectors) < self.args['var_threshold'])[0]

                for p in positions:
                    removed_rows.append(p)

            removed_rows = list(set(removed_rows))

            kdes = {}
            for label in tqdm(range(self.args['num_classes']), desc="kde"):

                refined_ats = np.transpose(self.train_ats[self.class_matrix[label]])
                refined_ats = np.delete(refined_ats, removed_rows, axis=0)

                if refined_ats.shape[0] == 0:
                    print(
                        "Ats were removed by threshold {}".format(self.args['var_threshold'])
                    )
                    break
                kdes[label] = gaussian_kde(refined_ats)

        else:
            row_vectors = np.transpose(self.train_ats)
            for activation_node in range(row_vectors.shape[0]):
                if np.var(row_vectors[activation_node]) < self.args['var_threshold']:
                    removed_rows.append(activation_node)

            refined_ats = np.transpose(self.train_ats)
            refined_ats = np.delete(refined_ats, removed_rows, axis=0)
            if refined_ats.shape[0] == 0:
                print("Ats were removed by threshold {}".format(self.args['var_threshold']))
            kdes = [gaussian_kde(refined_ats)]

        print("The number of removed columns: {}".format(len(removed_rows)))

        return kdes, removed_rows

    def _calc_lsa(self, target_ats, target_pred, kdes, removed_rows, ds_name):

        """
            Calculate scalar LSA value of target activation traces

                Args:
                    target_ats (ndarray): Activation traces of target_data.
                    target_pred(ndarray): 1-D Array of predicted labels
                    ds_name (str): Type of dataset: Test or Target.
                    removed_rows: Positions to skip
                    kdes: Dict of scipy kde objects

                Returns:
                    lsa (float): The scalar LSA value

        """

        lsa = []
        print("[" + ds_name + "] " + "Fetching LSA")

        if self.args['is_classification']:
            for i, at in enumerate(tqdm(target_ats)):
                label = target_pred[i]
                kde = kdes[label]
                refined_at = np.delete(at, removed_rows, axis=0)
                lsa.append(np.asscalar(-kde.logpdf(np.transpose(refined_at))))
        else:
            kde = kdes[0]
            for at in tqdm(target_ats):
                refined_at = np.delete(at, removed_rows, axis=0)
                lsa.append(np.asscalar(-kde.logpdf(np.transpose(refined_at))))

        return lsa

class DSA(SurpriseAdequacy):

    def calc(self, target_data: tf.data.Dataset, ds_name: str, use_cache = False):
        """
        Return DSA values for target. Note that target_data here means both test and adversarial data. Separate calls in main.

                        Args:
                            target_data (ndarray): x_test or x_target.
                            ds_name (str): Type of dataset: Train, Test, or Target.
                            use_cache (bool): Use stored files to load activation traces or not

                        Returns:
                            dsa (float): The scalar DSA value

        """
        target_ats, target_pred = self._load_or_calculate_ats(dataset=target_data, ds_name=ds_name, use_cache = use_cache)
        return self._calc_dsa(target_ats, target_pred, ds_name)

    def _calc_dsa(self, target_ats, target_pred, ds_name):

        """
        Calculate scalar DSA value of target activation traces

            Args:
                target_ats (ndarray): Activation traces of target_data.
                ds_name (str): Type of dataset: Test or Target.
                target_pred (ndarray): 1-D Array of predicted labels

            Returns:
                dsa (float): The scalar LSA value

        """


        dsa = np.empty(shape=target_pred.shape[0])
        batch_size = 500
        start = 0
        all_idx = list(range(len(self.train_pred)))

        print("[" + ds_name + "] " + "Fetching DSA")

        target_shape = target_pred.shape[0]
        while start < target_shape:
            diff = target_shape - start

            if diff < batch_size:
                batch = target_pred[start:start + diff]

            else:
                batch = target_pred[start: start + batch_size]

            for label in range(self.args['num_classes']):

                matches = np.where(batch == label)
                if len(matches) > 0:
                    target_matches = target_ats[matches[0] + start]
                    train_matches_sameClass = self.train_ats[self.class_matrix[label]]
                    a_dist = target_matches[:, None] - train_matches_sameClass
                    a_dist_norms = np.linalg.norm(a_dist, axis=2)
                    a_min_dist = np.min(a_dist_norms, axis=1)
                    closest_position = np.argmin(a_dist_norms, axis=1)
                    closest_ats = train_matches_sameClass[closest_position]

                    train_matches_otherClasses = self.train_ats[list(set(all_idx) - set(self.class_matrix[label]))]
                    b_dist = closest_ats[:, None] - train_matches_otherClasses
                    b_dist_norms = np.linalg.norm(b_dist, axis=2)
                    b_min_dist = np.min(b_dist_norms, axis=1)

                    dsa[matches[0] + start] = a_min_dist / b_min_dist

                else:
                    continue

            start += batch_size

        return dsa
