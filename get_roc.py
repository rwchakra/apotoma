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
import pandas as pd

root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"
datasets = ['fmnist', 'mnist-c', 'mnist-fgsm']


(_, _), (x_test_nominal, y_test_nominal) = mnist.load_data()
x_test_nominal = x_test_nominal.reshape(-1, 28, 28, 1)
x_test_nominal = (x_test_nominal / 255.0)
x_test_nominal = x_test_nominal.astype("float32")

for model_n in range(0, 10):
    model_score = np.empty((3, 3))
    for j,ds in enumerate(datasets):

        if ds == 'fmnist':
            #x_test_corrupted = np.load('mnist_corrupted.npy')
            (_, _), (x_test_corrupted, y_test_corrupted) = fashion_mnist.load_data()
            x_test_corrupted = x_test_corrupted.reshape(-1, 28, 28, 1)
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype("float32")

        elif ds == 'mnist-c':
            x_test_corrupted = np.load('mnist_corrupted.npy')
            # (_, _), (x_test_corrupted, y_test_corrupted) = fashion_mnist.load_data()
            x_test_corrupted = x_test_corrupted.reshape(-1, 28, 28, 1)
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype("float32")

        else:
            x_test_corrupted = np.load('ood_data/all_mnist_models/mnist_base_model_adv_'+str(model_n+1)+'.npy')
            # (_, _), (x_test_corrupted, y_test_corrupted) = fashion_mnist.load_data()
            x_test_corrupted = x_test_corrupted.reshape(-1, 28, 28, 1)
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype("float32")

        model = load_model('model/model/mnist_models/model_mnist_'+str(model_n+1)+'.h5')
        start = time.time()
        preds_corrupted = model.predict(x_test_corrupted)
        preds_nominal = model.predict(x_test_nominal)
        g = np.exp(preds_nominal - np.amax(preds_nominal, axis=1)[:, None])
        g = g / np.sum(g[:, None], axis=-1)
        f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
        f = f / np.sum(f[:, None], axis=-1)
        scores_corrupted = np.max(f, axis=1)
        scores_nominal = np.max(g, axis=1)

        y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
        y_true_nominal = np.zeros((x_test_nominal.shape[0]))

        y_true = np.concatenate([y_true_corrupted, y_true_nominal])
        y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1
        end = time.time()
        print("Time for Softmax: ", (end - start))
        print(roc_auc_score(y_true, y_scores))
        model_score[0][j] = roc_auc_score(y_true, y_scores)
        #Vanilla sofmax MNIST-C: 0.5985
        #Vanilla softmax MNIST-adv: 1.0

        args = argparse.Namespace()
        args.model_path = root+'/model/model/mnist_models/model_mnist_'+str(model_n+1)+'.h5'
        args.sub_model_path = root+'/submodels_dissector_latest/submodels_dissector_latest_v2/model_mnist/model_'+str(model_n+1)+'/'
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
        model_score[1][j] = roc_auc_score(y_true, y_scores)
        #DISSECTOR-Linear MNIST-C: 0.6144
        #DISSECTOR-Linear MNIST-Adv: 0.6218

        print("Running for OE....")
        model = load_model('model/model/mnist_models_finetuned/model_mnist_finetuned_'+str(model_n+1)+'.h5')


        if ds == 'mnist-fgsm':
            x_test_corrupted = np.load('ood_data/all_mnist_models_finetuned/mnist_base_model_finetuned_adv_'+str(model_n+1)+'.npy')
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype("float32")
            start = time.time()
            preds_corrupted = model.predict(x_test_corrupted)
            preds_nominal = model.predict(x_test_nominal)
            f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
            f = f / np.sum(f[:, None], axis=-1)
            g = np.exp(preds_nominal - np.amax(preds_nominal, axis=1)[:, None])
            g = g / np.sum(g[:, None], axis=-1)
            scores_corrupted = np.max(f, axis=1)
            scores_nominal = np.max(g, axis=1)

            y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
            y_true_nominal = np.zeros((x_test_nominal.shape[0]))

            y_true = np.concatenate([y_true_corrupted, y_true_nominal])
            y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1
            end = time.time()
            print(roc_auc_score(y_true, y_scores))
            print("Time for OE: ", (end - start))
            model_score[2][j] = roc_auc_score(y_true, y_scores)

        else:
            preds_corrupted = model.predict(x_test_corrupted)
            f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
            f = f / np.sum(f[:, None], axis=-1)
            preds_nominal = model.predict(x_test_nominal)
            g = np.exp(preds_nominal - np.amax(preds_nominal, axis=1)[:, None])
            g = g / np.sum(g[:, None], axis=-1)

            # preds_nominal = np.exp(preds_nominal - np.max(preds_nominal)) / np.sum(np.exp(preds_nominal - np.max(preds_nominal)), axis=-1, keepdims=True)
            scores_corrupted = np.max(f, axis=1)
            scores_nominal = np.max(g, axis=1)

            y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
            y_true_nominal = np.zeros((x_test_nominal.shape[0]))

            y_true = np.concatenate([y_true_corrupted, y_true_nominal])
            y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1

            end = time.time()
            print(roc_auc_score(y_true, y_scores))

            model_score[2][j] = roc_auc_score(y_true, y_scores)
            #OE MNIST-C: 0.8327
            #OE MNIST-ADV: 1.0

    df = pd.DataFrame(model_score)
    df.to_csv(root + '/model_' + str(model_n + 1) + '.csv', index=False)




