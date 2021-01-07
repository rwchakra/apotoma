import os
import pickle
import time
from typing import Dict, Tuple, List

import numpy as np
from dataclasses import dataclass
from sklearn import metrics

from apotoma.smart_dsa_diffnorms import DiffOfNormsSelectiveDSA
from apotoma.smart_dsa_normdiffs import NormOfDiffsSelectiveDSA
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
    # avg_pr_score: float
    accuracy: float
    ood_auc_roc: float
    num_nominal_samples: int
    num_outlier_samples: int
    # TODO maybe we can add a tpr at fpr.05 or something as well


def run_experiments(model,
                    sa_config: SurpriseAdequacyConfig,
                    train_x: np.ndarray,
                    test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Result]:
    results = []

    nominal_data = test_data.pop("nominal")

    for train_percent in range(5, 101, 5):
        num_samples = int(train_x.shape[0] * train_percent / 100)
        train_subset = train_x[:num_samples]
        # DSA
        dsa = DSA(model=model, train_data=train_subset, config=sa_config, dsa_batch_size=config.DSA_BATCH_SIZE)
        dsa_custom_info = {"sum_samples": num_samples, "dsa_batch_size": config.DSA_BATCH_SIZE}
        results.append(eval_for_sa(f"dsa_rand{train_percent}_perc", dsa, dsa_custom_info, nominal_data, test_data, ))
        # LSA
        lsa = LSA(model=model, train_data=train_subset, config=sa_config)
        lsa_custom_info = {"num_samples": num_samples}
        results.append(eval_for_sa(f"lsa_rand{train_percent}_perc", lsa, lsa_custom_info, nominal_data, test_data))

    # TODO We need to come up with dynamic way to calculate these
    for diff_threshold in [0.1, 0.001, 0.0001, .00001, 0.000001]:
        dsa = DiffOfNormsSelectiveDSA(model=model,
                                      train_data=train_x,
                                      config=sa_config,
                                      dsa_batch_size=config.DSA_BATCH_SIZE,
                                      threshold=diff_threshold)
        dsa_custom_info = {
            "diff_threshold": diff_threshold,
            "dsa_batch_size": config.DSA_BATCH_SIZE
        }
        results.append(eval_for_sa(f"dsa_don_{diff_threshold}", dsa, dsa_custom_info, nominal_data, test_data))

    # TODO We need to come up with dynamic way to calculate these
    for diff_threshold in range(10, 90, 10):
        diff_threshold /= 1000  # int percentage to float ratio
        dsa = NormOfDiffsSelectiveDSA(model=model,
                                      train_data=train_x,
                                      config=sa_config,
                                      dsa_batch_size=config.DSA_BATCH_SIZE,
                                      threshold=diff_threshold)
        dsa_custom_info = {
            "diff_threshold": diff_threshold,
            "dsa_batch_size": config.DSA_BATCH_SIZE
        }
        results.append(eval_for_sa(f"dsa_nod_{diff_threshold}", dsa, dsa_custom_info, nominal_data, test_data))

    return results


def eval_for_sa(sa_name,
                sa: SurpriseAdequacy,
                approach_custom_info: Dict,
                nominal_data: Tuple[np.ndarray, np.ndarray],
                test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Result:
    # Prepare SA (offline)
    prep_start_time = time.time()
    # TODO split DNN prediction and SA postprocessing (e.g. kde fitting)
    sa.prep(use_cache=USE_CACHE)
    prep_time = time.time() - prep_start_time
    if isinstance(sa, DiffOfNormsSelectiveDSA) or isinstance(sa, NormOfDiffsSelectiveDSA):
        approach_custom_info['num_samples'] = sa.number_of_samples

    # Create result object
    result = Result(name=sa_name, prepare_time=prep_time, approach_custom_info=approach_custom_info)

    nom_surp, nom_pred = sa.calc(target_data=nominal_data[0], use_cache=USE_CACHE, ds_type='test')

    for test_set_name, test_set in test_data.items():
        print(f"Evaluating {sa_name} with test set {test_set_name}")
        x_test = test_set[0]

        calc_start = time.time()
        # TODO split DNN prediction and SA calculation
        surp, pred = sa.calc(target_data=x_test, use_cache=USE_CACHE, ds_type='test')
        calc_time = time.time() - calc_start

        # Used for (outlier-only) misclassification prediction
        is_misclassified = test_set[1] != pred
        accuracy = (x_test.shape[0] - np.sum(is_misclassified)) / x_test.shape[0]
        # auc_roc = metrics.roc_auc_score(is_misclassified, surp)
        # avg_pr = metrics.average_precision_score(is_misclassified, surp)

        is_outlier = np.ones(shape=(nom_surp.shape[0] + surp.shape[0]), dtype=bool)
        is_outlier[:nom_surp.shape[0]] = 0
        combined_surp = np.concatenate((nom_surp, surp))
        ood_auc_roc = metrics.roc_auc_score(is_outlier, combined_surp)

        result.evals[test_set_name] = TestSetEval(eval_time=calc_time,
                                                  ood_auc_roc=ood_auc_roc,
                                                  accuracy = accuracy,
                                                  num_nominal_samples=nominal_data[0].shape[0],
                                                  num_outlier_samples=x_test.shape[0])

    return result


def save_results_to_fs(case_study: str, results: List[Result], model_id=int) -> None:
    for res in results:
        os.makedirs(f"../results/{case_study}/{res.name}", exist_ok=True)
        with open(f"../results/{case_study}/{res.name}/model_{model_id}.pickle", "wb+") as f:
            pickle.dump(res, file=f)
