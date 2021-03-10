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

def grad(model, inputs):
    #TODO tape.gradient needs to compute on loss, not softmax. Find a way without breaking the graph
  with tf.GradientTape() as tape:
    softmax, acts = forward_activations(model, inputs, training=True)
  return softmax, tape.gradient(softmax, model.trainable_variables), acts

def forward_activations(model, x, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    outputs = [layer.output for layer in model.layers]


    pre_softmax = outputs[-2]
    get_output = K.function([model.input],
                        [pre_softmax], K.set_learning_phase(1))
    [predictions_pre] = get_output([x])

    return y_, predictions_pre

def train(model, train_datagen):
    train_loss_results = []
    train_accuracy_results = []
    root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/'

    train_image_generator_out = train_datagen.flow_from_directory(
        root + 'ImageNet-Datasets-Downloader/dataset/imagenet_images/', batch_size=32, target_size=(32, 32))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0)
    x_test = (x_test / 255.0)

    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)
    train_image_generator_in = train_datagen.flow(x_train, y_train, batch_size=32)

    num_epochs = 10
    epoch_loss_avg = tf.keras.metrics.Mean()
    m = tf.keras.losses.CategoricalCrossentropy()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(num_epochs):

        epoch_loss = 0
        epoch_acc = 0
        # Training loop - using batches of 32
        for d_in, d_out in zip(train_image_generator_in, train_image_generator_out):
            data = tf.keras.layers.concatenate([d_in[0], d_out[0]], axis=0)
            target = d_in[1]
            # Below, instead of calling function, maybe just write the code here

            activations, grads, pre_softmax = grad(model, data)

            loss_value = m(activations[:len(d_in[0])], target).numpy()
            uniform_loss = tf.reduce_mean(tf.reduce_mean(pre_softmax[len(d_in[0]):], 1) - tf.reduce_logsumexp(pre_softmax[len(d_in[0]):], 1))
            loss_value += 0.5 * -uniform_loss.numpy()
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss += loss_value
            epoch_acc += epoch_accuracy([target], [activations[:len(d_in[0])]]).numpy()
            #epoch_loss_avg.update_state(K.constant(loss_value))  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            #epoch_accuracy.update_state([target], [activations[:len(d_in[0])]])

        # End epoch
        train_loss_results.append(np.mean(epoch_loss))
        train_accuracy_results.append(np.mean(epoch_acc))

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        np.mean(epoch_loss),
                                                                        np.mean(epoch_acc)))



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)





opt = Adam(learning_rate=1e-5)
root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/'
model = load_model(root+'model/model_outexp_cifar.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train(model, train_datagen)

