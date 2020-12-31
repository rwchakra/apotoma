import os
import pickle
import time
from typing import Dict, Tuple, List

import numpy as np
from dataclasses import dataclass
from sklearn import metrics

from apotoma.surprise_adequacy import LSA, DSA, SurpriseAdequacyConfig, SurpriseAdequacy
from case_studies import config

# We accept redundant at calculation for now - may change this later
USE_CACHE = False


class Result:
    def __init__(self,
                 name: str,
                 prepare_time: float,
                 approach_custom_info: Dict):
        self.name = name
        self.prepare_time = prepare_time
        self.approach_custom_info = approach_custom_info
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
                    test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Result]:

    results = []

    for train_percent in range(5, 101, 5):
        num_samples = int(train_x.shape[0] * train_percent / 100)
        train_subset = train_x[:num_samples]
        # DSA
        dsa = DSA(model=model, train_data=train_subset, config=sa_config, dsa_batch_size=config.DSA_BATCH_SIZE)
        dsa_custom_info = {"num_base_samples": num_samples, "dsa_batch_size": config.DSA_BATCH_SIZE}
        results.append(eval_for_sa(f"dsa_rand{train_percent}_perc", dsa, dsa_custom_info, test_data))
        # LSA
        lsa = LSA(model=model, train_data=train_subset, config=sa_config)
        lsa_custom_info = {"num_samples": num_samples}
        results.append(eval_for_sa(f"lsa_rand{train_percent}_perc", lsa, lsa_custom_info, test_data))

    return results


def eval_for_sa(sa_name,
                sa: SurpriseAdequacy,
                approach_custom_info: Dict,
                test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Result:
    # Prepare SA (offline)
    prep_start_time = time.time()
    # TODO split DNN prediction and SA postprocessing (e.g. kde fitting)
    sa.prep(use_cache=USE_CACHE)
    prep_time = time.time() - prep_start_time

    # Create result object
    result = Result(name=sa_name, prepare_time=prep_time, approach_custom_info=approach_custom_info)

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


def save_results_to_fs(case_study: str, results: List[Result]) -> None:
    os.makedirs(f"../results/{case_study}", exist_ok=True)
    for res in results:
        with open(f"results/{case_study}/{res.name}.pickle", "wb+") as f:
            pickle.dump(res, file=f)
