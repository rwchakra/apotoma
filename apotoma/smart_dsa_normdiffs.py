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

        new_class_matrix_norms_vec = {}
        for label in range(10):

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

                diffs = np.linalg.norm(candidates - data_label[current_idx], axis=1)
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
            new_class_matrix_norms_vec[label] = list(available_indices[selected_indexes])

        self.number_of_samples = sum(len(lst) for lst in new_class_matrix_norms_vec.values())

        self.class_matrix = new_class_matrix_norms_vec

    def sample_diff_distributions(self, x_subarray: np.ndarray) -> np.ndarray:
        ats, pred = self._calculate_ats(x_subarray)
        unique_pred = np.unique(pred)
        differences = []
        for label in unique_pred:
            label_indices = np.where(pred == label)
            values = ats[label_indices]
            diff_matrix = np.abs(values - np.expand_dims(values, 1))
            norms = np.linalg.norm(diff_matrix, axis=2)  # Norms
            indeces_under_diagonal = np.tril_indices(norms.shape[0], -1)
            diffs = norms[indeces_under_diagonal]
            differences += list(diffs)
        return np.array(sorted(differences))
