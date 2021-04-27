'''

1. Load MNIST-C and normal MNIST test set
2. Run model_mnist.evaluate and get max softmax value
3. Run dissector and get PV score on both MNIST nominal and MNIST-c
4. Run OE fine tuned model.evaluate on both MNIST and MNIST-c
'''

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
from sklearn.metrics import roc_auc_score
from dissector_temp_folder import dissector
import argparse
import time

root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"

x_test_corrupted = np.load('mnist_corrupted.npy')
#(_, _), (x_test_corrupted, y_test_corrupted) = fashion_mnist.load_data()
#x_test_corrupted = x_test_corrupted.reshape(-1, 28, 28, 1)
x_test_corrupted = (x_test_corrupted / 255.0)
x_test_corrupted = x_test_corrupted.astype("float32")

(_, _), (x_test_nominal, y_test_nominal) = mnist.load_data()
x_test_nominal = x_test_nominal.reshape(-1, 28, 28, 1)
x_test_nominal = (x_test_nominal / 255.0)
x_test_nominal = x_test_nominal.astype("float32")


model = load_model('model/model_mnist.h5')
start = time.time()
preds_corrupted = model.predict(x_test_corrupted)
preds_nominal = model.predict(x_test_nominal)
scores_corrupted = np.max(preds_corrupted, axis=1)
scores_nominal = np.max(preds_nominal, axis=1)

y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
y_true_nominal = np.zeros((x_test_nominal.shape[0]))

y_true = np.concatenate([y_true_corrupted, y_true_nominal])
y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1
end = time.time()
print("Time for Softmax: ", (end - start))
print(roc_auc_score(y_true, y_scores))
#Vanilla sofmax MNIST-C: 0.5985
#Vanilla softmax MNIST-adv: 1.0

args = argparse.Namespace()
args.model_path = root+'/model/model_mnist.h5'
args.sub_model_path = root+'/submodels_dissector_latest/model_mnist/'
diss = dissector.Dissector(model, config=args)
start = time.time()
sv_scores_corrupted = diss.sv_score(x_test_corrupted)
weights = diss.get_weights(growth_type='linear', alpha=0.9)

pv_scores_corrupted = diss.pv_scores(weights, sv_scores_corrupted)

sv_scores_nominal = diss.sv_score(x_test_nominal)
pv_scores_nominal = diss.pv_scores(weights, sv_scores_nominal)

y_scores = np.concatenate([pv_scores_corrupted, pv_scores_nominal]) * -1
end = time.time()
print(roc_auc_score(y_true, y_scores))
print("Time for dissector: ", (end - start))
#DISSECTOR-Linear MNIST-C: 0.6144
#DISSECTOR-Linear MNIST-Adv: 0.6218

model = load_model('model/model/mnist_models_finetuned/model_mnist_finetuned.h5')

#x_test_corrupted = np.load('ood_data/adversarial/mnist/mnist_base_model_finetuned_adv_1.npy')
#x_test_corrupted = (x_test_corrupted / 255.0)
#x_test_corrupted = x_test_corrupted.astype("float32")
start = time.time()
preds_corrupted = model.predict(x_test_corrupted)
preds_nominal = model.predict(x_test_nominal)
scores_corrupted = np.max(preds_corrupted, axis=1)
scores_nominal = np.max(preds_nominal, axis=1)

y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
y_true_nominal = np.zeros((x_test_nominal.shape[0]))

y_true = np.concatenate([y_true_corrupted, y_true_nominal])
y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1
end = time.time()
print(roc_auc_score(y_true, y_scores))
print("Time for OE: ", (end - start))
#OE MNIST-C: 0.8327
#OE MNIST-ADV: 1.0




