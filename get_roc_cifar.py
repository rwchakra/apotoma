'''

1. Load MNIST-C and normal MNIST test set
2. Run model_mnist.evaluate and get max softmax value
3. Run dissector and get PV score on both MNIST nominal and MNIST-c
4. Run OE fine tuned model.evaluate on both MNIST and MNIST-c
'''

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10
from sklearn.metrics import roc_auc_score
from dissector_temp_folder import dissector
import argparse

root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"

x_test_corrupted = np.load('ood_data/cifar10-c/corrupted_images_v2.npy')
x_test_corrupted = (x_test_corrupted / 255.0)
x_test_corrupted = x_test_corrupted.astype("float32")

(_, _), (x_test_nominal, y_test_nominal) = cifar10.load_data()
#x_test_nominal = x_test_nominal.reshape(-1, 28, 28, 1)
x_test_nominal = (x_test_nominal / 255.0)
x_test_nominal = x_test_nominal.astype("float32")


model = load_model('model/model_outexp_nosmcifar.h5')
print("Running for vanilla softmax...")
preds_nominal = model.predict(x_test_nominal)
scores_nominal = np.max(preds_nominal, axis=1)
# for i in range(5):
#     x_test_corrupted_intensity = x_test_corrupted[:, i]
#     preds_corrupted = model.predict(x_test_corrupted_intensity)
#
#     scores_corrupted = np.max(preds_corrupted, axis=1)
#
#
#     y_true_corrupted = np.ones((x_test_corrupted_intensity.shape[0]))
#     y_true_nominal = np.zeros((x_test_nominal.shape[0]))
#
#     y_true = np.concatenate([y_true_corrupted, y_true_nominal])
#     y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1
#
#     print(roc_auc_score(y_true, y_scores))
    #Vanilla sofmax MNIST-C: 0.5985
    #Vanilla softmax MNIST-adv: 1.0
    #Vanilla sofmax CIFAR10-C: 0.63
    #Vanilla softmax CIFAR10-adv: 0.91

# print("Running for Dissector...")
# args = argparse.Namespace()
# args.model_path = root+'/model/model_outexp_nosmcifar.h5'
# args.sub_model_path = root+'/submodels_dissector_latest/model_outexp_nosmcifar/'
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
#     print(roc_auc_score(y_true, y_scores))
# # #DISSECTOR-Linear MNIST-C: 0.6144
# # DISSECTOR-Linear MNIST-Adv: 0.6218
# # DISSECTOR-Linear CIFAR10-C: 0.622
# # DISSECTOR-Linear CIFAR10-Adv: 0.8685
#
model = load_model('model/model_outexp_nosmcifar_finetuned.h5')

#x_test_corrupted = np.load('ood_data/outexp_nosmcifar_finetune_adv.npy')
#x_test_corrupted = (x_test_corrupted / 255.0)
#x_test_corrupted = x_test_corrupted.astype("float32")

print("Running for OE....")
for i in range(5):
    x_test_corrupted_intensity = x_test_corrupted[:, i]

    preds_corrupted = model.predict(x_test_corrupted_intensity)
    preds_nominal = model.predict(x_test_nominal)
    scores_corrupted = np.max(preds_corrupted, axis=1)
    scores_nominal = np.max(preds_nominal, axis=1)

    y_true_corrupted = np.ones((x_test_corrupted_intensity.shape[0]))
    y_true_nominal = np.zeros((x_test_nominal.shape[0]))

    y_true = np.concatenate([y_true_corrupted, y_true_nominal])
    y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1

    print(roc_auc_score(y_true, y_scores))

#OE MNIST-C: 0.8231
#OE MNIST-ADV: 1.0
#OE CIFAR10-C: 0.72
#OE CIFAR10-ADV: 0.9507




