from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def grad(model, inputs):
  with tf.GradientTape() as tape:
    softmax, acts = forward_activations(model, inputs, training=True)
  return softmax, tape.gradient(softmax, model.trainable_variables), acts

def forward_activations(model, x, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    outputs = [layer.output for layer in model.layers]

    return y_, outputs

def train(train_image_generator_in, train_image_generator_out):
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(num_epochs):


        # Training loop - using batches of 32
        for d_in, d_out in zip(train_image_generator_in, train_image_generator_out):
            data = keras.layers.concatenate()[d_in, d_out]
            target = d_in[1]
            target = utils.to_categorical(target)
            # Optimize the model
            activations, grads, layer_activations = grad(model, data)
            pre_softmax = layer_activations[::-1]

            loss_value = tf.keras.metrics.CategoricalCrossentropy()(activations[:len(d_in)], target)
            loss_value += 0.5 * - tf.reduce_mean(tf.reduce_mean(pre_softmax[len(d_in[0]):], 1) - tf.reduce_logsumexp(pre_softmax[d_in[0]:], 1))
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(target, model(d_in, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_image_generator_out = train_datagen.flow_from_directory(
    root+'ImageNet-Datasets-Downloader/dataset/imagenet_images/', batch_size=32, target_size=(32, 32))

opt = Adam(learning_rate=1e-5)
model = load_model(root+'model/model_outexp_cifar.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

earlystopping = EarlyStopping(monitor = 'accuracy', verbose=1,
                              min_delta=0.01, patience=3, mode='max')

callbacks_list = [earlystopping]

