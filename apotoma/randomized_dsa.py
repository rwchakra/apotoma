import numpy as np
import tensorflow as tf

from apotoma.surprise_adequacy import DSA, SurpriseAdequacyConfig


class RandomizedDSA(DSA):

    def __init__(self,
                 model: tf.keras.Model,
                 train_data: np.ndarray,
                 config: SurpriseAdequacyConfig,
                 dsa_batch_size=500) -> None:
        super().__init__(model, train_data, config, dsa_batch_size)