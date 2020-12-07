import abc

import tensorflow as tf
import numpy as np


class NoveltyScore(abc.ABC):

    def __init__(self, model: tf.keras.Model, train_data: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.train_data = train_data

    # @abc.abstractmethod says that this method *must* be implemented
    #   in any child class for which we want to create instances
    @abc.abstractmethod
    def prep(self):
        pass

    @abc.abstractmethod
    def calc(self, target_data: np.ndarray, ds_name: str):
        pass

    @abc.abstractmethod
    def clear_cache(self, saved_path: str):
        pass
