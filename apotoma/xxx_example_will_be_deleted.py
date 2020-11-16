from random import random

from apotoma.surprise_adequacy import DSA, LSA

USE_DSA = False


def _auc_roc(sa, true_labels):
    print("Yeahh!")
    pass


if __name__ == '__main__':

    # 'Experimental Setup'
    model = None
    train_data = None
    test_data = None
    true_labels = None

    # Useage of Library
    novelty_score = None
    if USE_DSA:
        novelty_score = DSA(train_data=train_data, model=model)
    else:
        novelty_score = LSA(train_data=train_data, model=model)

    novelty_score.prep()
    sa = novelty_score.calc(test_data)

    # Evaluation
    _auc_roc(sa, true_labels)