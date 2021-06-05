from random import random

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from apotoma.surprise_adequacy import DSA, LSA
from apotoma.surprise_adequacy import  SurpriseAdequacyConfig
from evaluate import _auc_roc
from train_model import CLIP_MAX, CLIP_MIN
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import pandas as pd
import os

USE_DSA = False


if __name__ == '__main__':

    # 'Experimental Setup'
    model = None
    train_data = None
    test_data = None
    true_labels = None

    root = "/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma"

    args = {'d': 'mnist', 'is_classification': True,
            'dsa':False, 'lsa':True, 'batch_size': 128,
            'var_threshold': 1e-5, 'upper_bound': 2000,
            'n_bucket': 1000, 'num_classes': 10,
            'layer_names': ['activation_3'], 'saved_path': './tmp1/'}

    assert args['d'] in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args['lsa'] ^ args['dsa'], "Select either 'lsa' or 'dsa'"
    print(args)

    datasets = ['fmnist', 'mnist-c', 'mnist-fgsm']
    #datasets = ['cifar100', 'cifar-fgsm']
    (x_train, y_train), (x_test_nominal, y_test) = mnist.load_data()
    #(_, _), (x_test_nominal, y_test_nominal) = cifar10.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test_nominal = x_test_nominal.reshape(-1, 28, 28, 1)

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test_nominal = x_test_nominal.astype("float32")
    x_test_nominal= (x_test_nominal / 255.0) - (1.0 - CLIP_MAX)

    # models = os.listdir('./model/model/mnist_models')
    # for model in models:
    #     m = load_model('./model/model/mnist_models/'+model)
    #     print(m.layers[-1])

    model_score = np.empty((10, 3))
    model_times = []
    for model_n in range(0, 10):
        print("Running for model: ", model_n)
        model = load_model("./model/model/mnist_models/model_mnist_" + str(model_n+1) + ".h5")
        #model = load_model('model/model/cifar_models/model_outexp_nosmcifar_' + str(model_n) + '.h5')
        model.summary()
        # Load pre-trained model.
        for j, ds in enumerate(datasets):

            if ds == 'fmnist':
                (_, _), (x_adv, _) = fashion_mnist.load_data()
                x_adv = (x_adv / 255.0) - (1.0 - CLIP_MAX)
                x_adv = x_adv.astype("float32")

            elif ds == 'mnist-c':
                x_adv = np.load('mnist_corrupted.npy')
                #x_adv = np.load('ood_data/cifar10-c/corrupted_images_v2.npy')
                x_adv = (x_adv / 255.0) - (1.0 - CLIP_MAX)
                x_adv = x_adv.astype("float32")

            else:
                x_adv = np.load("ood_data/all_mnist_models/mnist_base_model_adv_"+str(model_n + 1)+".npy")
                #x_adv = np.load('ood_data/all_cifar_models/cifar_base_model_adv_' + str(model_n + 1) + '.npy')
                x_adv = (x_adv / 255.0) - (1.0 - CLIP_MAX)
                x_adv = x_adv.astype("float32")



            # Load target set. Fixed for now, will change later, added as an args element
            #x_test_corrupted = np.load("ood_data/adversarial/mnist/mnist_base_model_adv_1.npy")
            #(_, _), (x_test_corrupted, _) = fashion_mnist.load_data()
            #x_test_corrupted = np.load('ood_data/cifar10-c/corrupted_images_v2.npy')
            #(_, _), (x_test_corrupted, _) = cifar100.load_data()
            #x_test_corrupted = (x_test_corrupted / 255.0) - (1.0 - CLIP_MAX)
            #x_test_corrupted = x_test_corrupted.astype("float32")

            #(_, _), (x_adv, _) = fashion_mnist.load_data()
            #x_adv = x_adv.reshape(-1, 28, 28, 1)
            #x_adv = np.load("mnist_corrupted.npy")
            #x_adv = (x_adv / 255.0) - (1.0 - CLIP_MAX)
            #x_adv = x_adv.astype("float32")

            # preds_nominal = model.predict(x_test_nominal)
            # scores_nominal = np.max(preds_nominal, axis=1)
            # y_true_corrupted = np.ones((x_test_corrupted.shape[0]))
            # y_true_nominal = np.zeros((x_test_nominal.shape[0]))
            #
            # y_true = np.concatenate([y_true_corrupted, y_true_nominal])


            config = SurpriseAdequacyConfig(saved_path='./tmp1/',layer_names=[model.layers[-2].name], ds_name='mnist', num_classes=10,
                                            is_classification=True)
            # Usage of Library
            novelty_score = None
            if args['dsa']:
                novelty_score = DSA(train_data=x_train, model=model, config=config, max_workers=1)
            else:
                novelty_score = LSA(train_data=x_train, model=model, config=config)



            novelty_score.prep()

            if ds == 'cifar10-c':
                avg_sc = 0
                for i in range(5):
                    x_adv_intensity = x_adv[:, i]
                    start = time.time()
                    scores_test = novelty_score.calc(x_test_nominal, "test", use_cache=False)
                    scores_adv = novelty_score.calc(x_adv_intensity, "adv", use_cache=False)

                #y_scores_corr = np.concatenate([scores_corrupted[0], scores_nominal])
                #y_scores_adv = np.concatenate([scores_adv[0], scores_nominal])
                #print(roc_auc_score(y_true, y_scores_corr))
                #print(roc_auc_score(y_true, y_scores_adv))
                # Evaluation
                    score = _auc_roc(scores_test[0], scores_adv[0])
                    avg_sc += score
                    end = time.time()
                    print(score)
                    print("Time taken: ", (end - start))

                model_score[model_n][j] = avg_sc

            else:

                start = time.time()
                scores_test = novelty_score.calc(x_test_nominal, "test", use_cache=False)
                scores_adv = novelty_score.calc(x_adv, "adv", use_cache=False)
                end = time.time()
                score = _auc_roc(scores_test[0], scores_adv[0])

                print(score)
                print("Model: ", str(model_n + 1))
                print("Time taken: {}".format(ds), (end - start))
                print("Model times", model_times)
                model_score[model_n][j] = score
                model_times.append((end-start))
                print(np.mean(model_times))
                print(np.std(model_times))

        #df = pd.DataFrame(model_score)
        #df.to_csv(root+'/mnist_lsa_auc_'+str(model_n+1)+'.csv', index = False)

    np.save(root+'/lsa_mnist_times.npy', model_times)
