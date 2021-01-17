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

NUM_MODELS = 10


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
            0: 3,
            1: 3
        }

    @classmethod
    def gpu_memory_limit(cls) -> int:
        return 1500


class PredictContext(uwiz.models.ensemble_utils.DynamicGpuGrowthContextManager):

    @classmethod
    def max_sequential_tasks_per_process(cls) -> int:
        return 1


def train_model(model_id):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                                     input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', name="2nd_last_dense"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu', name="last_dense"))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name="sm_output"))

    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])

    return model, "history_not_returned"


def _get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    return x_train, y_train, x_test, y_test


def run_experiments(model_id, model):
    print(f"Starting with model id {model_id}")
    x_train, _, x_test, y_test = _get_dataset()

    # epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    advs = get_adv_data(model, x_test, y_test, epsilons=[0.5])
    corrupted = np.load(str(os.path.join(config.BASE_FOLDER, "datasets", "cifar10_corrupted.npy")))
    test_data = {
        'nominal': (x_test, y_test),
        'corrupted_sev_5': (corrupted[:, 4] / 255., y_test),
        # TODO change once doing multiple epsilons
        'adv_fga_0.5': (advs, y_test)
    }
    temp_folder = "/tmp/" + str(time.time())
    os.makedirs(temp_folder, exist_ok=True)
    sa_config = SurpriseAdequacyConfig(
        saved_path=temp_folder,
        is_classification=True,
        layer_names=["sm_output"],
        ds_name=f"cifar10_{model_id}",
        num_classes=10,
        batch_size=128)
    results = utils.run_experiments(model=model,
                                    train_x=x_train,
                                    test_data=test_data,
                                    sa_config=sa_config)
    utils.save_results_to_fs(results=results, case_study="cifar10", model_id=model_id)
    shutil.rmtree(temp_folder)


def get_adv_data(model, x_test, y_test, epsilons):
    badge_size = 500
    x_test = np.reshape(x_test, (-1, badge_size, 32, 32, 3))
    y_test = np.reshape(y_test, (-1, badge_size))

    adv = []
    for i in range(x_test.shape[0]):
        fmodel = foolbox.models.TensorFlowModel(model, bounds=(0, 1), device='CPU:0')
        attack = foolbox.attacks.LinfFastGradientAttack()
        attack_x = tf.convert_to_tensor(x_test[i])
        attack_y = tf.convert_to_tensor(y_test[i], dtype=tf.int32)
        advs, _, success = attack(fmodel, attack_x, attack_y, epsilons=epsilons)
        adv.append(advs[0].numpy())
        print(f"Completed foolbox batch {i}")
    return np.concatenate(adv).reshape((-1, 32, 32, 3))


if __name__ == '__main__':
    # Prepare dataset in cache
    # tf.keras.datasets.cifar10.load_data()

    model_collection = uwiz.models.LazyEnsemble(num_models=NUM_MODELS,
                                                model_save_path=config.MODELS_BASE_FOLDER + "cifar10",
                                                delete_existing=False,
                                                expect_model=True)
    # histories = model_collection.create(
    #     train_model, num_processes=6, context=TrainContext
    # )

    model_collection.consume(
        run_experiments, num_processes=1, context=PredictContext
    )
