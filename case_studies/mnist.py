import json
import os
import shutil
import time
from typing import Dict

import foolbox
import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

from apotoma.surprise_adequacy import SurpriseAdequacyConfig
from case_studies import config, utils

NUM_MODELS = 20


class TrainContext(uwiz.models.ensemble_utils.DeviceAllocatorContextManager):

    @classmethod
    def file_path(cls) -> str:
        return "temp-ensemble.txt"

    @classmethod
    def run_on_cpu(cls) -> bool:
        return False

    @classmethod
    def virtual_devices_per_gpu(cls) -> Dict[int, int]:
        return {
            0: 4,
            1: 4
        }

    @classmethod
    def gpu_memory_limit(cls) -> int:
        return 1000

def run_experiments(model_id, model):
    x_train, _, x_test, y_test = _get_dataset()

    # epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    advs = get_adv_data(model, x_test, y_test, epsilons=[0.5])
    test_data = {
        'nominal': (x_test, y_test),
        #TODO change once doing multiple epsilons
        'adv_fga_0.5': (advs[0].numpy(), y_test) # TODO maybe mix with nominal data?
    }
    temp_folder = "/tmp/" + str(time.time())
    os.mkdir(temp_folder)
    sa_config = SurpriseAdequacyConfig(
        saved_path=temp_folder,
        is_classification=True,
        layer_names=["sm_output"],
        ds_name=f"mnist_{model_id}",
        num_classes=10)
    results = utils.run_experiments(model=model,
                                    train_x=x_train,
                                    test_data=test_data,
                                    sa_config=sa_config)
    utils.save_results_to_fs(results=results, case_study="mnist")
    shutil.rmtree(temp_folder)


def get_adv_data(model, x_test, y_test, epsilons):
    fmodel = foolbox.models.TensorFlowModel(model, bounds=(0, 1))
    attack = foolbox.attacks.LinfFastGradientAttack()
    attack_x = tf.convert_to_tensor(x_test)
    attack_y = tf.convert_to_tensor(y_test, dtype=tf.int32)
    advs, _, success = attack(fmodel, attack_x, attack_y, epsilons=epsilons)
    return advs


def train_model(model_id):
    """
    Trains an mnist model. According to https://keras.io/examples/vision/mnist_convnet/, but with an additional layer.
    :param model_id:
    :return:
    """
    import tensorflow as tf
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, name="last_dense"),
            tf.keras.layers.Dense(10, activation="softmax", name="sm_output"),
        ]
    )

    x_train, y_train, _, _ = _get_dataset()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

    return model, history.history


def _get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # Prepare dataset in cache
    tf.keras.datasets.mnist.load_data()

    model_collection = uwiz.models.LazyEnsemble(num_models=NUM_MODELS,
                                                model_save_path=config.MODELS_BASE_FOLDER + "mnist",
                                                delete_existing=False)
    # histories = model_collection.create(
    #     train_model, num_processes=8, context=TrainContext
    # )

    model_collection.consume(
        run_experiments, num_processes=20
    )
