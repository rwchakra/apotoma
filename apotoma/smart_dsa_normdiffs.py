import os

import numpy as np
import tensorflow as tf

from apotoma.surprise_adequacy import DSA, SurpriseAdequacyConfig


class NormOfDiffsSelectiveDSA(DSA):

    def __init__(self,
                 model: tf.keras.Model,
                 train_data: np.ndarray,
                 config: SurpriseAdequacyConfig,
                 threshold=0.05,
                 dsa_batch_size=500) -> None:
        super().__init__(model, train_data, config, dsa_batch_size)
        self.threshold = threshold

    def _load_or_calc_train_ats(self, use_cache=False) -> None:

        saved_train_path = self._get_saved_path("train")

        if os.path.exists(saved_train_path[0]) and use_cache:
            print("Found saved {} ATs, skip serving".format("train"))
            # In case train_ats is stored in a disk
            self.train_ats, self.train_pred = np.load(saved_train_path[0]), np.load(saved_train_path[1])

        else:
            self.train_ats, self.train_pred = self._load_or_calculate_ats(dataset=self.train_data, ds_type="train",
                                                                          use_cache=use_cache)

        self._select_smart_ats()

    def prep(self, use_cache: bool = False) -> None:
        self._load_or_calc_train_ats(use_cache=use_cache)

    def _select_smart_ats(self):

        all_train_ats = self.train_ats
        all_train_pred = self.train_pred

        class_pred_matrix = {}

        # TODO change this to work for regression and general number of classes
        # collected_ats = []
        # collected_pred = []
        for label in range(10):
            # ats = all_train_ats[all_train_pred == label]
            # is_available_mask = np.ones(dtype=bool, shape=ats.shape[0])
            #
            # class_pred_matrix[label] = []
            #
            # i = 0
            # while True:
            #     current_ats = ats[i]
            #     # Select with (all_train_ats) index i in the selected list of ats
            #     # and put its new index in the new matrix
            #     class_pred_matrix[label].append(len(collected_ats)) # The new index
            #     collected_ats.append(current_ats)
            #     collected_pred.append(label)  # For regression: replace with all_train_pred[i]
            #
            #     # Current ats is selected and becomes unavailable
            #     is_available_mask[i] = False
            #
            #     # Calculate differences and update is_available_mask
            #     avail_ats = ats[is_available_mask]
            #     diffs = np.linalg.norm(avail_ats - current_ats, axis=1)
            #     is_available_indexes = np.where(is_available_mask)[0]
            #     drop_indeces = is_available_indexes[np.where(diffs < 0)] # TODO switch to thresholds
            #     is_available_mask[drop_indeces] = False
            #
            #     i = np.argmax(is_available_mask)
            #     if i == 0:
            #         break
            #

            # OLD CODE
            data_label = all_train_ats[np.where(all_train_pred == label)]
            available_indices = np.where(all_train_pred == label)[0]

            indexes = np.arange(available_indices.shape[0])
            is_available = np.ones(shape=available_indices.shape[0], dtype=bool)

            # This index used in the loop indicates the latest element selected to be added to chosen items
            current_idx = 0

            while True:
                # Get all indexes (higher than current_index) which are still available and the corresponding ats
                candidate_indexes = np.argwhere((indexes > current_idx) & is_available).flatten()
                candidates = data_label[candidate_indexes]

                diffs = np.linalg.norm(np.abs(candidates - data_label[current_idx]), axis=1)
                assert diffs.ndim == 1

                # Identify candidates which are too similar to currently added element (current_idx)
                # and set their availability to false
                remove_candidate_indexes = np.flatnonzero(diffs < self.threshold)
                remove_overall_indexes = candidate_indexes[remove_candidate_indexes]
                is_available[remove_overall_indexes] = False

                # Select the next available candidate as current_idx (i.e., use select it for use in dsa),
                #   or break if none available
                if np.count_nonzero(is_available[current_idx:]) > 1:
                    current_idx = np.argmax(is_available[current_idx + 1:]) + (current_idx + 1)
                else:
                    break

            selected_indexes = np.nonzero(is_available)[0]
            class_pred_matrix[label] = list(available_indices[selected_indexes])

            # TODO Move to proper unit test
            # selected_predictions = all_train_pred[class_pred_matrix[label]]
            # assert np.std(selected_predictions) == 0, "Not all predictions are for the same class"
            # selected_ats = all_train_ats[class_pred_matrix[label]]
            # # selected_ats = np.array(collected_ats)[np.array(collected_pred) == label]
            # for i in range(selected_ats.shape[0] - 1):
            #     # Note: This completely ignores labels
            #     min_dist = np.min(np.linalg.norm(selected_ats[1 + i:] - selected_ats[i], axis=1))
            #     assert min_dist >= self.threshold, f"Found difference {min_dist} < {self.threshold}"
            # # END TEST

        self.number_of_samples = sum(len(lst) for lst in class_pred_matrix.values())

        self.class_matrix = class_pred_matrix

    def sample_diff_distributions(self, x_subarray: np.ndarray, num_samples=100) -> np.ndarray:
        """
        Calculates all differences between the samples passed in the subarray.
        This can be used to guess thresholds for the algorithm.
        The threshold passed when creating this DSA instance is ignored.
        :param x_subarray: the subset of the train data (or any other data) for which to calc the differences
        :return: Sorted one-dimensional array of differences
        """
        ats, pred = self._calculate_ats(x_subarray)
        differences = np.empty(shape=num_samples)
        for i in range(num_samples):
            # Note: This completely ignores labels
            min_dist = np.min(np.linalg.norm(np.abs(ats[1 + i:] - ats[i]), axis=1))
            differences[i] = min_dist
        return np.sort(differences)
