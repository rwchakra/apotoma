import abc

import tensorflow as tf


class NoveltyScore(abc.ABC):

    def __init__(self, model: tf.keras.Model, train_data: tf.data.Dataset) -> None:
        super().__init__()
        self.model = model
        self.train_data = train_data

    # @abc.abstractmethod sais that this method *must* be implemented
    #   in any child class for which we want to create instances
    @abc.abstractmethod
    def prep(self):
        pass

    @abc.abstractmethod
    def calc(self, target_data: tf.data.Dataset):
        pass
