from typing import Dict

import tensorflow as tf
import uncertainty_wizard as uwiz

from case_studies import config

NUM_MODELS = 200


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


def train_model(model_id):
    import tensorflow as tf

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
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu', name="for_sa"))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # For the sake of this example, let's use just one epoch.
    # Of course, for higher accuracy, you should use more.
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])

    return model, "history_not_returned"


if __name__ == '__main__':
    # Prepare dataset in cache
    tf.keras.datasets.cifar10.load_data()

    model_collection = uwiz.models.LazyEnsemble(num_models=NUM_MODELS,
                                                model_save_path=config.MODELS_BASE_FOLDER + "cifar10",
                                                delete_existing=False)
    histories = model_collection.create(
        train_model, num_processes=6, context=TrainContext
    )
