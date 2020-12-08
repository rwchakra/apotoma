import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from apotoma.surprise_adequacy import DSA, SurpriseAdequacyConfig


class SmartDSA(DSA):

    def __init__(self,
                 model: tf.keras.Model,
                 train_data: np.ndarray,
                 config: SurpriseAdequacyConfig,
                 number_of_samples: int,
                 dsa_batch_size=500) -> None:
        super().__init__(model, train_data, config, dsa_batch_size)
        self.number_of_samples = number_of_samples

    def _load_or_calc_train_ats(self, use_cache=False) -> None:
        smart_paths = self._get_saved_path(ds_type=f'smart_train_subset_{self.number_of_samples}')
        if use_cache and os.path.exists(smart_paths[0]):
            self.train_ats = np.load(smart_paths[0])
            self.train_pred = np.load(smart_paths[1])
        else:
            self._load_or_select_smart_ats(smart_paths=smart_paths, use_cache=use_cache)

    def _load_or_select_smart_ats(self, smart_paths: Tuple[str, str], use_cache: bool):
        if self.train_ats.shape[0] > self.number_of_samples:
            super()._load_or_calc_train_ats(use_cache=use_cache)
            self._select_smart_ats()
            if use_cache:
                np.save(smart_paths[0], self.train_ats)
                np.save(smart_paths[1], self.train_pred)
                print(f"Saved the smart train ats selection and predictions to {smart_paths[0]} and {smart_paths[1]}")
        else:
            raise UserWarning((f"Configured SmartDSA to select the activations of the {self.number_of_samples} "
                               f"training samples, but only {self.train_ats.shape[0]} "
                               f"training samples were passed. Thus, SmartDSA will continue as regular DSA."))

    def _select_smart_ats(self):
        all_train_ats = self.train_ats
        all_train_pred = self.train_pred

        # TODO @Rwiddhi  *** Here comes your super smart algorithm **
        #   (smartly select the `self.number_of_samples` many best train ats
        #   and store them to the local fields)

        self.train_ats = None  # TODO set selected_train_ats_here
        self.train_pred = None  # TODO set pred of selected train_ats here
