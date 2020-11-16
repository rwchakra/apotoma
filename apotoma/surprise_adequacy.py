from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf

from apotoma.novelty_score import NoveltyScore


class SurpriseAdequacy(NoveltyScore):

    def __init__(self, model: tf.keras.Model, train_data: tf.data.Dataset) -> None:
        super().__init__(model, train_data)
        self.train_ats, self.train_pred = None, None

    # Returns ats and returns predictions
    def _calculate_ats(self, dataset: tf.data.Dataset, ds_name: str) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Assume we calculated the ats for {ds_name} dataset")
        # Do something like self.model.predict(dataset)
        if ds_name == "train":
            print("Assume we cache the train ats and predictions here")

    # TODO: Think about (long term): Maybe no caching as we keep this im memory anyways
    def _get_or_calc_train_ats(self):
        is_chached = False  # TODO Replace with call to see if cached arrays exist
        if is_chached:
            return None, None  # TODO return loaded arrays
        else:
            self.train_ats, self.train_pred = self._calculate_ats(dataset=self.train_data, ds_name="train")
            # TODO cache to FS

    def prep(self):
        self.train_ats, self.train_pred = self._get_or_calc_train_ats()


class LSA(SurpriseAdequacy):

    def calc(self, target_data: tf.data.Dataset):
        target_ats, target_pred = self._calculate_ats(dataset=target_data, ds_name="target")
        kdes, removed_rows = self._calc_kdes(target_ats)
        return self._calc_lsa(target_ats, target_pred, kdes, removed_rows)

    def _calc_kdes(self, target_ats) -> Tuple[Dict[int, None], List[int]]:  # Replace None with kde scipy object type
        # TODO calc kdes and removed_rows based on self.train_ats, kdes
        pass

    def _calc_lsa(self, target_ats, target_pred, kdes, removed_rows):
        # TODO calc lsa (log stuff)
        pass


# TODO Implement Class DSA
class DSA(SurpriseAdequacy):
    pass
