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
import pandas as pd

root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"
datasets = ['cifar100','cifar-fgsm']

(_, _), (x_test_nominal, y_test_nominal) = cifar10.load_data()
#x_test_nominal = x_test_nominal.reshape(-1, 28, 28, 1)
x_test_nominal = (x_test_nominal / 255.0)
x_test_nominal = x_test_nominal.astype("float32")
model_times_sm = []
model_times_diss = []
model_times_oe = []
times = {'diss': [], 'oe': [], 'sm': []}
for model_n in range(0, 10):
    model_score = np.zeros((3, 3))
    for j, ds in enumerate(datasets):
        if ds == 'cifar100':
            #x_test_corrupted = np.load('ood_data/cifar10-c/corrupted_images_v2.npy')
            (_, _), (x_test_corrupted, y_test_corrupted) = cifar100.load_data()
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype("float32")

        elif ds == 'cifar10-c':
            x_test_corrupted = np.load('ood_data/cifar10-c/corrupted_images_v2.npy')
            #(_, _), (x_test_corrupted, y_test_corrupted) = cifar100.load_data()
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype("float32")

        else:
            x_test_corrupted = np.load('ood_data/all_cifar_models/cifar_base_model_adv_'+str(model_n+1)+'.npy')
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype('float32')

        model = load_model('model/model/cifar_models/model_outexp_nosmcifar_'+str(model_n)+'.h5')
        print("Running for vanilla softmax...")
        preds_nominal = model.predict(x_test_nominal)
        g = np.exp(preds_nominal - np.amax(preds_nominal, axis=1)[:, None])
        g = g/np.sum(g[:, None], axis=-1)
        #preds_nominal = np.exp(preds_nominal) / np.sum(np.exp(preds_nominal), axis=-1, keepdims=True)
        scores_nominal = np.max(g, axis=1)

        if ds == 'cifar10-c':
            avg_sc = 0
            for i in range(5):
                x_test_corrupted_intensity = x_test_corrupted[:, i]
                preds_corrupted = model.predict(x_test_corrupted_intensity)
                f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
                f = f/np.sum(f[:, None], axis=-1)
                #preds_corrupted = np.exp(preds_nominal) / np.sum(np.exp(preds_nominal), axis=-1, keepdims=True)
                scores_corrupted = np.max(f, axis=1)


                y_true_corrupted = np.ones((x_test_corrupted_intensity.shape[0]))
                y_true_nominal = np.zeros((x_test_nominal.shape[0]))

                y_true = np.concatenate([y_true_corrupted, y_true_nominal])
                y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1

                end = time.time()
                avg_sc += roc_auc_score(y_true, y_scores)
                print(roc_auc_score(y_true, y_scores))
            model_score[0][j] = avg_sc/5


        else:
            start = time.time()
            preds_corrupted = model.predict(x_test_corrupted)
            f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
            f = f / np.sum(f[:, None], axis=-1)
            # preds_corrupted = np.exp(preds_nominal) / np.sum(np.exp(preds_nominal), axis=-1, keepdims=True)
            scores_corrupted = np.max(f, axis=1)
            end = time.time()
            y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
            y_true_nominal = np.zeros((x_test_nominal.shape[0]))

            y_true = np.concatenate([y_true_corrupted, y_true_nominal])
            y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1


            print(roc_auc_score(y_true, y_scores))
            model_score[0][j] = roc_auc_score(y_true, y_scores)
        #     print("Time taken for Softmax: {}".format(ds), (end - start))
            model_times_sm.append((end-start))
        # # #     #Vanilla sofmax MNIST-C: 0.5985
        # # #     #Vanilla softmax MNIST-adv: 1.0
        # # #     #Vanilla sofmax CIFAR10-C: 0.63
        # # #     #Vanilla softmax CIFAR10-adv: 0.91
        # # #
        print("Running for Dissector...")
        start = time.time()
        args = argparse.Namespace()
        args.model_path = root+'/model/model/cifar_models/model_outexp_nosmcifar_'+str(model_n)+'.h5'
        args.sub_model_path = root+'/submodels_dissector_latest/submodels_dissector_latest_v2/model_outexp_nosmcifar/model_'+str(model_n+1)+'/'

        if ds == 'cifar10-c':
            avg_sc = 0
            for i in range(5):
                x_test_corrupted_intensity = x_test_corrupted[:, i]
                diss = dissector.Dissector(model, config=args)

                sv_scores_corrupted = diss.sv_score(x_test_corrupted_intensity)
                weights = diss.get_weights(growth_type='linear', alpha=0.9)

                pv_scores_corrupted = diss.pv_scores(weights, sv_scores_corrupted)

                sv_scores_nominal = diss.sv_score(x_test_nominal)
                pv_scores_nominal = diss.pv_scores(weights, sv_scores_nominal)

                y_true_corrupted = np.ones((x_test_corrupted_intensity.shape[0]))
                y_true_nominal = np.zeros((x_test_nominal.shape[0]))
                y_true = np.concatenate([y_true_corrupted, y_true_nominal])

                y_scores = np.concatenate([pv_scores_corrupted, pv_scores_nominal]) * -1
                end = time.time()
                avg_sc += roc_auc_score(y_true, y_scores)
                print(roc_auc_score(y_true, y_scores))
                print("Time for Dissector: ", (end - start))

            model_score[1][j] = avg_sc/5

        else:


            diss = dissector.Dissector(model, config=args)
            start = time.time()

            sv_scores_corrupted = diss.sv_score(x_test_corrupted)
            weights = diss.get_weights(growth_type='linear', alpha=0.9)

            pv_scores_corrupted = diss.pv_scores(weights, sv_scores_corrupted)

            sv_scores_nominal = diss.sv_score(x_test_nominal)
            pv_scores_nominal = diss.pv_scores(weights, sv_scores_nominal)
            end = time.time()
            y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
            y_true_nominal = np.zeros((x_test_nominal.shape[0]))
            y_true = np.concatenate([y_true_corrupted, y_true_nominal])

            y_scores = np.concatenate([pv_scores_corrupted, pv_scores_nominal]) * -1

            print(roc_auc_score(y_true, y_scores))
            print("Time for Dissector: {}".format(ds), (end - start))
            model_score[1][j] = roc_auc_score(y_true, y_scores)
            model_times_diss.append((end-start))
        #
        # # # # #DISSECTOR-Linear MNIST-C: 0.6144
        # # # # DISSECTOR-Linear MNIST-Adv: 0.6218
        # # # # DISSECTOR-Linear CIFAR10-C: 0.622
        # # # # DISSECTOR-Linear CIFAR10-Adv: 0.8685
        # # #
        print("Running for OE....")
        model = load_model('model/model/cifar_models_finetuned/model_outexp_nosmcifar_finetuned_'+str(model_n)+'.h5')

        if ds == 'cifar-fgsm':
            x_test_corrupted = np.load('ood_data/all_cifar_models_finetuned/cifar_base_model_finetuned_adv_'+str(model_n+1)+'.npy')
            x_test_corrupted = (x_test_corrupted / 255.0)
            x_test_corrupted = x_test_corrupted.astype("float32")


        #for i in range(5):
        #x_test_corrupted_intensity = x_test_corrupted[:, i]
            start = time.time()
            preds_corrupted = model.predict(x_test_corrupted)
            f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
            f = f/np.sum(f[:, None], axis=-1)
            end = time.time()
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


            print(roc_auc_score(y_true, y_scores))
            print("Time for OE: {}".format(ds), (end - start))

            model_score[2][j] = roc_auc_score(y_true, y_scores)
            model_times_oe.append((end-start))

        elif ds =='cifar10-c':
            avg_sc = 0
            for i in range(5):
                x_test_corrupted_intensity = x_test_corrupted[:, i]

                preds_corrupted = model.predict(x_test_corrupted_intensity)
                f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
                f = f / np.sum(f[:, None], axis=-1)
                preds_nominal = model.predict(x_test_nominal)
                g = np.exp(preds_nominal - np.amax(preds_nominal, axis=1)[:, None])
                g = g / np.sum(g[:, None], axis=-1)

                # preds_nominal = np.exp(preds_nominal - np.max(preds_nominal)) / np.sum(np.exp(preds_nominal - np.max(preds_nominal)), axis=-1, keepdims=True)
                scores_corrupted = np.max(f, axis=1)
                scores_nominal = np.max(g, axis=1)

                y_true_corrupted = np.ones((x_test_corrupted_intensity.shape[0]))
                y_true_nominal = np.zeros((x_test_nominal.shape[0]))

                y_true = np.concatenate([y_true_corrupted, y_true_nominal])
                y_scores = np.concatenate([scores_corrupted, scores_nominal]) * -1

                end = time.time()
                avg_sc += roc_auc_score(y_true, y_scores)
                print(roc_auc_score(y_true, y_scores))
                print("Time for OE: ", (end - start))

            model_score[2][j] = avg_sc/5

        else:
            start = time.time()
            preds_corrupted = model.predict(x_test_corrupted)
            f = np.exp(preds_corrupted - np.amax(preds_corrupted, axis=1)[:, None])
            f = f / np.sum(f[:, None], axis=-1)
            end =  time.time()
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


            print(roc_auc_score(y_true, y_scores))
            print("Time for OE: {}".format(ds), (end - start))
            model_score[2][j] = roc_auc_score(y_true, y_scores)
            model_times_oe.append((end-start))

    print("Model: ", str(model_n + 1))
    print("SM times", model_times_sm)
    print("Diss times", model_times_diss)
    print("OE times", model_times_oe)
    #df = pd.DataFrame(model_score)
    #df.to_csv(root+'/cifar_auc_'+str(model_n+1)+'.csv', index = False)

times['oe'] = model_times_oe
times['sm'] = model_times_sm
times['diss'] = model_times_diss

np.save(root+'/model_times_cifar.npy', times)
#print(np.mean(model_times))
#print(np.std(model_times))




