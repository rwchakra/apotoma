import numpy as np
import png
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist, cifar100
import os
import random
(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = np.load('ood_data/adversarial/mnist/mnist_base_model_adv_1.npy')
labels  = np.load('ood_data/adversarial/mnist/mnist_base_model_adv_1_labels.npy')
#print(labels[0])
print(y_test[0])

zeros_adv = data[np.where(labels == 0)]
#print(np.shape(zeros_adv))
#zeros = x_test[np.where(y_test == 0)]
#plt.imshow(np.squeeze(zeros_adv[10]))
plt.imsave('ood_data/mnist_adv_0.5.png', (np.squeeze(zeros_adv[10])))
#plt.show()

# label = 6
# corruptions = os.listdir('./ood_data/mnist_c')
#
# for corr in corruptions:
#     data = np.load('./ood_data/mnist_c/' + corr + '/' + 'test_images.npy')
#     data_labels = np.load('./ood_data/mnist_c/' + corr + '/' + 'test_labels.npy')
#     six = np.where(data_labels == label)[0]
#     img = data[six[0]]
#     plt.imsave('ood_data/mnist_c_{}.png'.format(corr), np.squeeze(img))

# corruptions = np.load('ood_data/cifar10-c/all_corruptions_v2.npy', allow_pickle=True)
# images = np.load('ood_data/cifar10-c/corrupted_images_v2.npy', allow_pickle=True)
# l = np.load('ood_data/cifar10-c/labels.npy', allow_pickle=True)
#
# corrs = ['Gaussian Noise', 'Shot Noise', 'Impulse Noise', 'Defocus Blur', 'Glass Blur', 'Motion Blur',
#          'Zoom Blur', 'Snow', 'Frost', 'Fog', 'Brightness', 'Contrast',
#          'Elastic', 'Pixelate', 'JPEG']
#
# fig = plt.figure(figsize=(256,256))
# ax = plt.axes()
# for c in corrs:
#     i = np.where(corruptions == c)[0]
#
#     img = images[i[0]][4]
#     ax.imshow(img, interpolation='nearest')
#     plt.imsave('ood_data/cifar_c_{}.png'.format(c),img)

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#
# labels = range(0, 10)
# for label in labels:
#     im = np.where(y_test == label)[0][0]
#     plt.imsave('ood_data/fashion_mnist_{}.png'.format(label), x_test[im])
#     #plt.show()

# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
#
# labels = random.sample(range(0, 100), 10)
#
# for label in labels:
#     im = np.where(y_test == label)[0][0]
#     plt.imsave('ood_data/cifar100_images/cifar100_{}.png'.format(label), x_test[im])
#     #plt.show()