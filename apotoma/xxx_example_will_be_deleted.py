from random import random

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from apotoma.surprise_adequacy import DSA, LSA
from evaluate import _auc_roc

import numpy as np

USE_DSA = False


if __name__ == '__main__':

    # 'Experimental Setup'
    model = None
    train_data = None
    test_data = None
    true_labels = None

    args = {'d': 'mnist', 'is_classification': True,
            'dsa': False, 'lsa': True, 'batch_size': 128,
            'var_threshold': 1e-5, 'upper_bound': 2000,
            'n_bucket': 1000, 'num_classes': 10,
            'layer_names': ['activation_3'], 'saved_path': './tmp/'}

    assert args['d'] in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args['lsa'] ^ args['dsa'], "Select either 'lsa' or 'dsa'"
    print(args)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Load pre-trained model.
    model = load_model("./model/model_mnist.h5")
    model.summary()


    # Load target set. Fixed for now, will change later, added as an args element
    #x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))
    x_target = np.load("./adv/adv_mnist_fgsm.npy")

    # Usage of Library
    novelty_score = None
    if args['dsa']:
        novelty_score = DSA(train_data=x_train, model=model, args=args)
    else:
        novelty_score = LSA(train_data=train_data, model=model, args=args)

    novelty_score.prep()
    sa_test = novelty_score.calc(x_test, "test")
    sa_target = novelty_score.calc(x_target, "target")

    # Evaluation
    score = _auc_roc(sa_test, sa_target)
    print(score)