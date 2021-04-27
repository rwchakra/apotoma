import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import foolbox
import matplotlib.pyplot as plt

batch_size = 500

for m in range(1, 10):
    model = load_model('model/model/cifar_models_finetuned/model_outexp_nosmcifar_finetuned_{}.h5'.format(m))
    root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32")
    x_test = x_test / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    x_test = np.reshape(x_test, (-1, batch_size, 32, 32, 3))
    y_test = np.reshape(y_test, (-1, batch_size))

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

    #test = advs[10]

    #plt.imshow(test)
    #plt.show()
    #plt.imsave('ood_data/mnist_adv_0.0.png', np.squeeze(test))
    np.save(root+'/ood_data/all_cifar_models_finetuned/cifar_base_model_finetuned_adv_{}'.format(m+1), advs)
    np.save(root+'/ood_data/all_cifar_models_finetuned/cifar_base_model_finetuned_adv_{}_labels'.format(m+1), adv_labels)
    print("Completed model ", m)
