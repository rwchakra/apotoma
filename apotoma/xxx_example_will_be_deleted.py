from random import random

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from apotoma.surprise_adequacy import DSA, LSA, SurpriseAdequacyConfig
from apotoma.smart_dsa_normdiffs import SmartDSA
from evaluate import _auc_roc
from train_model import CLIP_MAX, CLIP_MIN

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
            'layer_names': ['activation_3'], 'saved_path': './tmp1/'}

    assert args['d'] in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args['lsa'] ^ args['dsa'], "Select either 'lsa' or 'dsa'"
    print(args)

    config = SurpriseAdequacyConfig(is_classification=True, ds_name='mnist', layer_names=['activation_3']
                                    , saved_path='/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/tmp1', num_classes=10)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    # Load pre-trained model.
    model = load_model("/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/model/model_mnist.h5")
    model.summary()


    # Load target set. Fixed for now, will change later, added as an args element
    #x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))
    x_target = np.load("/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/adv/adv_mnist_fgsm.npy")

    # Usage of Library
    '''if args['dsa']:
        novelty_score = DSA(train_data=x_train, model=model, args=args)
    else:
        novelty_score = LSA(train_data=x_train, model=model, args=args)'''

    novelty_score = SmartDSA(model=model, config=config, train_data=x_train, number_of_samples=100)



    novelty_score.prep()
    sa_test = novelty_score.calc(x_test, "test", use_cache=False)[0]
    sa_target = novelty_score.calc(x_target, "target", use_cache=False)[0]

    # Evaluation
    score = _auc_roc(sa_test, sa_target)
    print(score)

    # Optional, clear cache
    #novelty_score.clear_cache(args['saved_path'])