from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

root = '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_image_generator = train_datagen.flow_from_directory(
    root+'ImageNet-Datasets-Downloader/dataset/imagenet_images/', batch_size=32, target_size=(32, 32))

opt = Adam(learning_rate=1e-5)
model = load_model(root+'model/model_outexp_cifar.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

earlystopping = EarlyStopping(monitor = 'accuracy', verbose=1,
                              min_delta=0.01, patience=3, mode='max')

callbacks_list = [earlystopping]

results = model.fit(train_image_generator, epochs=5,
                          callbacks=callbacks_list)
