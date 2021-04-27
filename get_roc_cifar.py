'''

1. Load MNIST-C and normal MNIST test set
2. Run model_mnist.evaluate and get max softmax value
3. Run dissector and get PV score on both MNIST nominal and MNIST-c
4. Run OE fine tuned model.evaluate on both MNIST and MNIST-c
'''

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from dissector_temp_folder import dissector
import argparse
import time

root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"


#x_test_corrupted = np.load('ood_data/cifar10-c/corrupted_images_v2.npy')
(_, _), (x_test_corrupted, y_test_corrupted) = cifar100.load_data()
x_test_corrupted = (x_test_corrupted / 255.0)
x_test_corrupted = x_test_corrupted.astype("float32")

(_, _), (x_test_nominal, y_test_nominal) = cifar10.load_data()
#x_test_nominal = x_test_nominal.reshape(-1, 28, 28, 1)
x_test_nominal = (x_test_nominal / 255.0)
x_test_nominal = x_test_nominal.astype("float32")


model = load_model('model/model/cifar_models/model_outexp_nosmcifar_1.h5')
print("Running for vanilla softmax...")
start = time.time()
preds_nominal = model.predict(x_test_nominal)
g = np.exp(preds_nominal - np.amax(preds_nominal, axis=1)[:, None])
g = g/np.sum(g[:, None], axis=-1)
#preds_nominal = np.exp(preds_nominal) / np.sum(np.exp(preds_nominal), axis=-1, keepdims=True)
scores_nominal = np.max(g, axis=1)

#for i in range(5):
#x_test_corrupted_intensity = x_test_corrupted[:, i]
preds_corrupted = model.predict(x_test_corrupted)
f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
f = f/np.sum(f[:, None], axis=-1)
#preds_corrupted = np.exp(preds_nominal) / np.sum(np.exp(preds_nominal), axis=-1, keepdims=True)
scores_corrupted = np.max(f, axis=1)


y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
y_true_nominal = np.zeros((x_test_nominal.shape[0]))

y_true = np.concatenate([y_true_corrupted, y_true_nominal])
y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1

end = time.time()
print(roc_auc_score(y_true, y_scores))
print("Time taken for Softmax: ", (end - start))
# #     #Vanilla sofmax MNIST-C: 0.5985
# #     #Vanilla softmax MNIST-adv: 1.0
# #     #Vanilla sofmax CIFAR10-C: 0.63
# #     #Vanilla softmax CIFAR10-adv: 0.91
# #
# print("Running for Dissector...")
# start = time.time()
# args = argparse.Namespace()
# args.model_path = root+'/model/model_outexp_nosmcifar.h5'
# args.sub_model_path = root+'/submodels_dissector_latest/submodels_dissector_latest_v2/model_outexp_nosmcifar/model_1/'
#
# for i in range(5):
#     x_test_corrupted_intensity = x_test_corrupted[:, i]
#     diss = dissector.Dissector(model, config=args)
#
#     sv_scores_corrupted = diss.sv_score(x_test_corrupted_intensity)
#     weights = diss.get_weights(growth_type='linear', alpha=0.9)
#
#     pv_scores_corrupted = diss.pv_scores(weights, sv_scores_corrupted)
#
#     sv_scores_nominal = diss.sv_score(x_test_nominal)
#     pv_scores_nominal = diss.pv_scores(weights, sv_scores_nominal)
#
#     y_true_corrupted = np.ones((x_test_corrupted_intensity.shape[0]))
#     y_true_nominal = np.zeros((x_test_nominal.shape[0]))
#     y_true = np.concatenate([y_true_corrupted, y_true_nominal])
#
#     y_scores = np.concatenate([pv_scores_corrupted, pv_scores_nominal]) * -1
#     end = time.time()
#     print(roc_auc_score(y_true, y_scores))
#     print("Time for Dissector: ", (end - start))

# # # #DISSECTOR-Linear MNIST-C: 0.6144
# # # DISSECTOR-Linear MNIST-Adv: 0.6218
# # # DISSECTOR-Linear CIFAR10-C: 0.622
# # # DISSECTOR-Linear CIFAR10-Adv: 0.8685
# #
model = load_model('model/model/cifar_models_finetuned/model_outexp_nosmcifar_finetuned_1.h5')

#x_test_corrupted = np.load('ood_data/all_cifar_models_finetuned/cifar_base_model_finetuned_adv_1.npy')
#x_test_corrupted = (x_test_corrupted / 255.0)
#x_test_corrupted = x_test_corrupted.astype("float32")

print("Running for OE....")
start = time.time()
#for i in range(5):
#x_test_corrupted_intensity = x_test_corrupted[:, i]

preds_corrupted = model.predict(x_test_corrupted)
f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
f = f/np.sum(f[:, None], axis=-1)
preds_nominal = model.predict(x_test_nominal)
g = np.exp(preds_nominal - np.amax(preds_nominal, axis=1)[:, None])
g = g/np.sum(g[:, None], axis=-1)

#preds_nominal = np.exp(preds_nominal - np.max(preds_nominal)) / np.sum(np.exp(preds_nominal - np.max(preds_nominal)), axis=-1, keepdims=True)
scores_corrupted = np.max(f, axis=1)
scores_nominal = np.max(g, axis=1)

y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
y_true_nominal = np.zeros((x_test_nominal.shape[0]))

y_true = np.concatenate([y_true_corrupted, y_true_nominal])
y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1

end = time.time()
print(roc_auc_score(y_true, y_scores))
print("Time for OE: ", (end - start))



