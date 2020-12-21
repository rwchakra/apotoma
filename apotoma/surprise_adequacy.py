import os
from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from typing import Tuple, List, Union, Dict

import numpy as np
import tensorflow as tf
from scipy.stats import gaussian_kde
from tensorflow.keras.models import Model
from tqdm import tqdm


@dataclass
class SurpriseAdequacyConfig:
    """Stores basic immutable surprise adequacy configuration.
    Instances of this class are reusable amongst different instances of surprise adequacy.

    Note: Jetbrains 'unresolved reference' is wrong: https://youtrack.jetbrains.com/issue/PY-28549

    Args:
        is_classification (bool): A boolean indicating if the NN under test solves a classification problem.
        num_classes (None, int): The number of classes (for classification problems)
        or None (for regression problems). Default: None

        layer_names (List(str)): List of layer names whose ATs are to be extracted. Code takes last layer.
        saved_path (str): Path to store and load ATs
        dataset_name (str): Dataset to be used. Currently supports mnist and cifar-10.
        num_classes (int): No. of classes in classification. Default is 10.
        min_var_threshold (float): Threshold value to check variance of ATs
        batch_size (int): Batch size to use while predicting.

     Raises:
        ValueError: If any of the config parameters takes an illegal value.
    """

    saved_path: str
    is_classification: bool
    layer_names: List[str]
    ds_name: str
    num_classes: Union[int, None]
    min_var_threshold: float = 1e-5
    batch_size: int = 128

    def __post_init__(self):
        if self.is_classification and not self.num_classes:
            raise ValueError("num_classes is a mandatory parameter "
                             "in SurpriseAdequacyConfig for classification problems")
        elif not self.is_classification and self.num_classes:
            raise ValueError(f"num_classes must be None (but was {self.num_classes}) "
                             "in SurpriseAdequacyConfig for classification problems")
        elif self.num_classes < 0 and self.is_classification:
            raise ValueError(f"num_classes must be positive but was {self.num_classes}) ")
        elif self.min_var_threshold < 0:
            raise ValueError(f"Variance threshold cannot be negative, but was {self.min_var_threshold}")

        elif self.ds_name is None or self.ds_name == "":
            raise ValueError(f"dataset name must not be None or empty")

        elif len(self.layer_names) == 0:
            raise ValueError(f"Layer list cannot be empty")
        elif len(self.layer_names) != len(set(self.layer_names)):
            raise ValueError(f"Layer list cannot contain duplicates")


class SurpriseAdequacy(ABC):

    def __init__(self, model: tf.keras.Model, train_data: np.ndarray, config: SurpriseAdequacyConfig) -> None:
        self.model = model
        self.train_data = train_data
        self.train_ats = None
        self.train_pred = None
        self.class_matrix = {}
        self.config = config

    def _get_saved_path(self, ds_type: str) -> Tuple[str, str]:
        """Determine saved path of ats and pred

        Args:
            ds_type: Type of dataset: Typically one of {Train, Test, Target}.

        Returns:
            ats_path: File path of ats.
            pred_path: File path of pred (independent of layers)
        """

        joined_layer_names = "_".join(self.config.layer_names)

        return (
            os.path.join(
                self.config.saved_path,
                self.config.ds_name + "_" + ds_type + "_" + joined_layer_names + "_ats" + ".npy",
            ),
            os.path.join(self.config.saved_path, self.config.ds_name + "_" + ds_type + "_pred" + ".npy"),
        )

    # Returns ats and returns predictions
    def _load_or_calculate_ats(self, dataset: np.ndarray, ds_type: str, use_cache: bool) -> Tuple[
        np.ndarray, np.ndarray]:

        """Determine activation traces train, target, and test datasets

        Args:
            dataset (ndarray): x_train or x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            ats (ndarray): Activation traces (Shape of num_examples * num_nodes).
            pred (ndarray): 1-D Array of predictions

        """
        print(f"Calculating the ats for {ds_type} dataset")

        saved_target_path = self._get_saved_path(ds_type)
        if saved_target_path is not None and os.path.exists(saved_target_path[0]) and use_cache:
            print(f"Found saved {ds_type} ATs, skip at collection from model")
            return self._load_ats(ds_type)
        else:
            ats, pred = self._calculate_ats(dataset)

            if saved_target_path is not None:
                np.save(saved_target_path[0], ats)
                np.save(saved_target_path[1], pred)
                print(f"[{ds_type}] Saved the ats and predictions to {saved_target_path[0]} and {saved_target_path[1]}")

            return ats, pred

    def _calculate_ats(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        temp_model = Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(layer_name).output for layer_name in self.config.layer_names]
        )

        if self.config.is_classification:
            # TODO Issue #20 (Michael) I think we can replace the following two predict statements to a single one
            #   By adding the original output layer to the outputs of the temp model.
            #   I will change that in a separate PR (and only after the tests are running),
            #   To make sure I do not mess anything up

            # Make the prediction on the original model.
            pred = np.argmax(self.model.predict(dataset, batch_size=self.config.batch_size, verbose=1),
                             axis=1)
            # Get the activation traces of the inner layers of the model.
            layer_outputs: list = temp_model.predict(dataset, batch_size=self.config.batch_size, verbose=1)

            # If we only collect the ats of a sinlge layer, they are not represented as list
            #   For compatibility, we pack them in a single-item list
            if len(self.config.layer_names) == 1:
                layer_outputs = [layer_outputs]

            ats = None
            for layer_name, layer_output in zip(self.config.layer_names, layer_outputs):
                print("Layer: " + layer_name)
                if layer_output[0].ndim == 3:
                    # For convolutional layers, taken over from original SA implementation
                    layer_matrix = np.array(
                        # TODO @Rwiddhi (issue 21): Here we have a loop over the dataset.
                        #   Check if this can be replaced using some numpy fun.
                        map(lambda x: [np.mean(x[..., j]) for j in range(x.shape[-1])],
                            [layer_output[i] for i in range(len(dataset))])
                    )
                else:
                    layer_matrix = np.array(layer_output)

                if ats is None:
                    # Shape of ats will be num_inputs x num_nodes_in_layer
                    ats = layer_matrix
                else:
                    ats = np.append(ats, layer_matrix, axis=1)

        return ats, pred

    def _load_ats(self, ds_type: str) -> Tuple[np.ndarray, np.ndarray]:
        # In case train_ats is stored in a disk
        saved_target_path = self._get_saved_path(ds_type)
        ats: np.ndarray = np.load(saved_target_path[0])
        pred: np.ndarray = np.load(saved_target_path[1])
        return ats, pred

    def _load_or_calc_train_ats(self, use_cache=False) -> None:
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
            self.train_ats, self.train_pred = self._load_or_calculate_ats(dataset=self.train_data, ds_type="train",
                                                                          use_cache=use_cache)

    def prep(self) -> None:
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

    def clear_cache(self, saved_path: str) -> None:
        """

        Delete files of activation traces.

        Args:
            saved_path(str): Base directory path

        """
        to_remove = ['train', 'test', 'target']
        for f in to_remove:
            path = self._get_saved_path(f)
            os.remove(os.path.join(saved_path, path[0]))
            os.remove(os.path.join(saved_path, path[1]))


        # files = [f for f in os.listdir(saved_path) if f.endswith('.npy')]
        # for f in files:
        #     os.remove(os.path.join(saved_path, f))


class LSA(SurpriseAdequacy):

    def calc(self, target_data: np.ndarray, ds_type: str, use_cache=False) -> List[float]:
        """
        Return LSA values for target. Note that target_data here means both test and adversarial data. Separate calls in main.

        Args:
            target_data (ndarray): x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            lsa (float): List of scalar LSA values

        """
        target_ats, target_pred = self._load_or_calculate_ats(dataset=target_data, ds_type=ds_type, use_cache=use_cache)

        kdes, removed_rows = self._calc_kdes()
        return self._calc_lsa(target_ats, target_pred, kdes, removed_rows, ds_type)

    def _calc_kdes(self) -> Tuple[dict, List[int]]:
        """
        Determine Gaussian KDE for each label and list of removed rows based on variance threshold, if any.

        Args:
            target_data (ndarray): x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            kdes: Dict - labels are keys, values are scipy kde objects
            removed_rows: Array of positions of removed rows

        """

        if self.config.is_classification:
            kdes, removed_rows = self._classification_kdes()
        else:
            kdes, removed_rows = self._regression_kdes()

        print((f"Ignoring the activations of {len(removed_rows)} traces "
               f"as their variance is not high enough."))

        return kdes, removed_rows

    def _regression_kdes(self) -> Tuple[List[gaussian_kde], List[int]]:
        removed_rows = []
        row_vectors = np.transpose(self.train_ats)
        for activation_node in range(row_vectors.shape[0]):
            if np.var(row_vectors[activation_node]) < self.config.min_var_threshold:
                removed_rows.append(activation_node)
        refined_ats = np.transpose(self.train_ats)
        refined_ats = np.delete(refined_ats, removed_rows, axis=0)
        if refined_ats.shape[0] != 0:

            kdes = [gaussian_kde(refined_ats)]
            return kdes, removed_rows

        else:
            raise ValueError(f"All ats were removed by threshold: ", self.config.min_var_threshold)

    def _classification_kdes(self) -> Tuple[Dict[int, gaussian_kde], List[int]]:
        removed_rows = []
        for label in range(self.config.num_classes):
            # Shape of (num_activation nodes x num_examples_by_label)
            row_vectors: np.ndarray = np.transpose(self.train_ats[self.class_matrix[label]])
            positions: np.ndarray = np.where(np.var(row_vectors) < self.config.min_var_threshold)[0]

            for p in positions:
                removed_rows.append(p)
        removed_rows = list(set(removed_rows))

        kdes = {}
        for label in tqdm(range(self.config.num_classes), desc="kde"):

            refined_ats = np.transpose(self.train_ats[self.class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_rows, axis=0)

            if refined_ats.shape[0] == 0:
                print(f"Ats for label {label} were removed by threshold {self.config.min_var_threshold}")
                break

            kdes[label] = gaussian_kde(refined_ats)

        return kdes, removed_rows

    def _calc_lsa(self,
                  target_ats: np.ndarray,
                  target_pred: np.ndarray,
                  kdes: {},
                  removed_rows: list,
                  ds_type: str) -> List[float]:
        """
        Calculate scalar LSA value of target activation traces

        Args:
            target_ats (ndarray): Activation traces of target_data.
            target_pred(ndarray): 1-D Array of predicted labels
            ds_type (str): Type of dataset: Test or Target.
            removed_rows (list): Positions to skip
            kdes: Dict of scipy kde objects

        Returns:
            lsa (float): List of scalar LSA values

        """
        print(f"[{ds_type}] Calculating LSA")

        if self.config.is_classification:
            lsa: List[float] = self._calc_classification_lsa(kdes, removed_rows, target_ats, target_pred)
        else:
            lsa: List[float] = self._calc_regression_lsa(kdes, removed_rows, target_ats)
        return lsa

    @staticmethod
    def _calc_regression_lsa(kdes: {},
                             removed_rows: list,
                             target_ats: np.ndarray) -> List[float]:
        lsa = []
        kde = kdes[0]
        for at in tqdm(target_ats):
            refined_at: np.ndarray = np.delete(at, removed_rows, axis=0)
            lsa.append(np.asscalar(-kde.logpdf(np.transpose(refined_at))))
        return lsa

    @staticmethod
    def _calc_classification_lsa(kdes: {},
                                 removed_rows: list,
                                 target_ats: np.ndarray,
                                 target_pred: np.ndarray) -> List[float]:
        lsa = []
        for i, at in enumerate(tqdm(target_ats)):
            label = target_pred[i]
            kde = kdes[label]
            refined_at: np.ndarray = np.delete(at, removed_rows, axis=0)
            lsa.append(np.asscalar(-kde.logpdf(np.transpose(refined_at))))
        return lsa


class DSA(SurpriseAdequacy):

    def __init__(self, model: tf.keras.Model,
                 train_data: np.ndarray,
                 config: SurpriseAdequacyConfig,
                 dsa_batch_size=500) -> None:
        super().__init__(model, train_data, config)
        self.dsa_batch_size = dsa_batch_size

    def calc(self, target_data: np.ndarray, ds_type: str, use_cache=False) -> np.ndarray:
        """
        Return DSA values for target. Note that target_data here means both test and adversarial data. Separate calls in main.

        Args:
            target_data (ndarray): x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            dsa (float): List of scalar DSA values

        """
        target_ats, target_pred = self._load_or_calculate_ats(dataset=target_data, ds_type=ds_type, use_cache=use_cache)
        return self._calc_dsa(target_ats, target_pred, ds_type)

    def _calc_dsa(self, target_ats: np.ndarray, target_pred: np.ndarray, ds_type: str) -> np.ndarray:

        """
        Calculate scalar DSA value of target activation traces

        Args:
            target_ats (ndarray): Activation traces of target_data.
            ds_type (str): Type of dataset: Test or Target.
            target_pred (ndarray): 1-D Array of predicted labels

        Returns:
            dsa (float): List of scalar DSA values

        """

        dsa = np.empty(shape=target_pred.shape[0])
        start = 0
        all_idx = list(range(len(self.train_pred)))

        print(f"[{ds_type}] Calculating DSA")

        num_targets = target_pred.shape[0]
        while start < num_targets:

            # Select batch
            diff = num_targets - start
            if diff < self.dsa_batch_size:
                batch = target_pred[start:start + diff]
            else:
                batch = target_pred[start: start + self.dsa_batch_size]

            # Calculate DSA per label
            for label in range(self.config.num_classes):
                matches = np.where(batch == label)
                if len(matches) > 0:
                    a_min_dist, b_min_dist = self._dsa_distances(all_idx, label, matches, start, target_ats)
                    dsa[matches[0] + start] = a_min_dist / b_min_dist

            start += self.dsa_batch_size

        return dsa

    def _dsa_distances(self, all_idx: list, label: int, matches: np.ndarray, start: int, target_ats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        target_matches = target_ats[matches[0] + start]
        train_matches_same_class = self.train_ats[self.class_matrix[label]]
        a_dist = target_matches[:, None] - train_matches_same_class
        a_dist_norms = np.linalg.norm(a_dist, axis=2)
        a_min_dist = np.min(a_dist_norms, axis=1)
        closest_position = np.argmin(a_dist_norms, axis=1)
        closest_ats = train_matches_same_class[closest_position]
        train_matches_other_classes = self.train_ats[list(set(all_idx) - set(self.class_matrix[label]))]
        b_dist = closest_ats[:, None] - train_matches_other_classes
        b_dist_norms = np.linalg.norm(b_dist, axis=2)
        b_min_dist = np.min(b_dist_norms, axis=1)

        return a_min_dist, b_min_dist
