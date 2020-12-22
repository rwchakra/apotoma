import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn import metrics

from apotoma.surprise_adequacy import LSA, DSA, SurpriseAdequacyConfig, SurpriseAdequacy
from case_studies import config

# We accept redundant at calculation for now - may change this later
USE_CACHE = False


class Result:
    def __init__(self,
                 name: str,
                 prepare_time: float):
        self.name = name
        self.prepare_time = prepare_time
        self.evals: Dict[str, 'TestSetEval'] = dict()


@dataclass
class TestSetEval:
    eval_time: float
    avg_pr_score: float
    auc_roc: float
    # TODO maybe we can add a tpr at fpr.05 or something as well


def run_experiments(model,
                    sa_config: SurpriseAdequacyConfig,
                    train_x: np.ndarray,
                    test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    sas = dict()

    for train_percent in range(5, 101, 5):
        num_samples = int(train_x.shape[0] * train_percent / 100)
        train_subset = train_x[:num_samples]
        sas[f"dsa_{train_percent}"] = DSA(model=model, train_data=train_subset, config=sa_config, dsa_batch_size=config.DSA_BATCH_SIZE)
        sas[f"lsa_{train_percent}"] = LSA(model=model, train_data=train_subset, config=sa_config)

    # TODO add our own implementations here

    return [eval_for_sa(n, sa, test_data) for n, sa in sas.items()]


def eval_for_sa(sa_name,
                sa: SurpriseAdequacy,
                test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Result:
    # Prepare SA (offline)
    prep_start_time = time.time()
    # TODO split DNN prediction and SA postprocessing (e.g. kde fitting)
    sa.prep(use_cache=USE_CACHE)
    prep_time = time.time() - prep_start_time

    # Create result object
    result = Result(name=sa_name, prepare_time=prep_time)

    for test_set_name, test_set in test_data.items():
        print(f"Evaluating {sa_name} with test set {test_set_name}")
        x_test = test_set[0]

        calc_start = time.time()
        # TODO split DNN prediction and SA calculation
        surp, pred = sa.calc(target_data=x_test, use_cache=USE_CACHE, ds_type='test')
        calc_time = time.time() - calc_start

        is_misclassified = test_set[1] != pred
        auc_roc = metrics.roc_auc_score(is_misclassified, surp)
        avg_pr = metrics.average_precision_score(is_misclassified, surp)
        result.evals[test_set_name] = TestSetEval(eval_time=calc_time, auc_roc=auc_roc, avg_pr_score=avg_pr)

    return result
