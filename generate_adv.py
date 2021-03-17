import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import foolbox

badge_size = 500
model = load_model('model/model_outexp_nosmcifar_finetuned.h5')
root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = np.reshape(x_test, (-1, badge_size, 32, 32, 3))
y_test = np.reshape(y_test, (-1, badge_size))

adv = []
adv_labels = []
for i in range(x_test.shape[0]):
    fmodel = foolbox.models.TensorFlowModel(model, bounds=(0, 1), device='CPU:0')
    attack = foolbox.attacks.LinfFastGradientAttack()
    attack_x = tf.convert_to_tensor(x_test[i])
    attack_y = tf.convert_to_tensor(y_test[i], dtype=tf.int32)
    advs, _, success = attack(fmodel, attack_x, attack_y, epsilons=[0.5])
    adv.append(advs[0].numpy())
    adv_labels.extend(y_test[i])
    print(f"Completed foolbox batch {i}")

advs = np.concatenate(adv).reshape((-1, 32, 32, 3))
np.save(root+'/ood_data/outexp_nosmcifar_finetune_adv.npy', advs)
np.save(root+'/ood_data/outexp_nosmcifar_finetune_adv_labels.npy', adv_labels)
