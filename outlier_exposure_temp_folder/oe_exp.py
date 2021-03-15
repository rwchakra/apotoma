import tqdm
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.special import softmax as sf
import numpy as np

def train(model, train_datagen):

    train_loss_results = []
    train_accuracy_results = []
    root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0)
    # x_test = (x_test / 255.0)

    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    train_image_generator_out = tf.data.Dataset.from_generator(lambda: train_datagen.flow_from_directory(
        root + '/ImageNet-Datasets-Downloader/dataset_new/imagenet_images', batch_size=32, target_size=(32, 32)),
        output_types= (tf.float32, tf.float32))

    train_image_generator_in = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

    #train_image_generator_in = train_datagen.flow(x_train, y_train, batch_size=64, shuffle=False)
    num_epochs = 10
    epoch_loss_avg = tf.keras.metrics.Mean()
    m = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    for epoch in range(num_epochs):

        epoch_loss = []
        epoch_acc = []
        loss_avg = 0.0
        step = int(x_train.shape[0] / 32)
        # Training loop - using batches of 32
        for d_in, d_out in tqdm.tqdm(zip(train_image_generator_in, train_image_generator_out), total=step):
            data = tf.keras.layers.concatenate([d_in[0], d_out[0]], axis=0)
            target = d_in[1]
            step -= 1
            if step == 0:
                break
            # Below, instead of calling function, maybe just write the code here
            with tf.GradientTape() as tape:
                pre_softmax = model(data, training=True)
                loss_value = m(target, pre_softmax[:len(d_in[0])])
                uniform_loss = tf.reduce_mean(
                tf.reduce_mean(pre_softmax[len(d_in[0]):], 1) - tf.reduce_logsumexp(pre_softmax[len(d_in[0]):], 1))
                loss_value = tf.keras.layers.Add()([loss_value, -0.5*uniform_loss])

            grads = tape.gradient(loss_value, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            loss_avg = loss_avg * 0.8 + float(loss_value) * 0.2

            epoch_loss.append(loss_avg)
        # End epoch
        train_loss_results.append(np.mean(epoch_loss))
        #train_accuracy_results.append(np.mean(epoch_acc))

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch,np.mean(epoch_loss)))



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

opt = Adam(learning_rate=1e-5)

root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"
model = load_model(root+'/model/model_outexp_nosmcifar.h5')

# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])


train(model, train_datagen)

