import numpy as np
import os
import random

corruptions = os.listdir('./mnist_c')

cifar_c, labels = [], []
all_corruptions = []
all_labels = {}
c = 0
mnist_c = np.empty(shape=(10000, 28, 28, 1), dtype=np.uint8)

for i in range(10000):
    corruption = random.choice(corruptions)
    data = np.load('./mnist_c/'+corruption+'/'+'test_images.npy')[i]
    mnist_c[i] = data
    c+=1
    print(c)
np.save('mnist_corrupted.npy', mnist_c)