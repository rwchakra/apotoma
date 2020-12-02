# TODO in this folder we will implement integration tests.
#   Integration tests test larger workflows than unit tests and may thus take longer to implement.
#   Integration tests are typically executed only when merging into main or selectively during development.


import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

from apotoma.surprise_adequacy import DSA
from apotoma.surprise_adequacy import LSA
from apotoma.dissector import Dissector

import argparse


class TestSurpriseAdequacyConsistency(unittest.TestCase):

    def setUp(self) -> None:
        self.model: tf.keras.Model = load_model(
            '/tests/assets/model_mnist.h5')
        (self.train_data, _), (self.test_data, y_test) = mnist.load_data()
        self.train_data = self.train_data.reshape(-1, 28, 28, 1)
        self.test_data = self.test_data.reshape(-1, 28, 28, 1)

        self.train_data = self.train_data.astype("float32")
        self.train_data = (self.train_data / 255.0) - (1.0 - 0.5)
        self.test_data = self.test_data.astype("float32")
        self.test_data = (self.test_data / 255.0) - (1.0 - 0.5)

    # DO this first
    def test_train_ats_calculation_against_kims_implementation(self, config):
        datasplit_train, datasplit_test = self.train_data[0:100], self.test_data[0:100]

        # HERE you'll calculate the ats on your code
        nodes = 10
        sa = DSA(self.model, datasplit_train, config=config)
        ats, pred = sa._calculate_ats()

        # Here you load the values from kims implementation
        kim_ats = np.load('./assets/mnist_train_activation_3_ats.npy')

        kim_pred = np.load('./assets/mnist_train_pred.npy')

        self.assertIsInstance(ats, np.ndarray)
        self.assertEqual(ats.shape, (100, nodes))
        self.assertEqual(ats.dtype, np.float32)
        np.testing.assert_almost_equal(ats, kim_ats[0:100], decimal=5)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred, (100,))
        self.assertEqual(pred.dtype, np.int)
        np.testing.assert_equal(pred, kim_pred)

    def test_dsa_is_consistent_with_original_implementation(self, config):

        our_dsa = DSA(model=self.model, train_data=self.train_data, config=config)
        our_dsa.prep()
        test_dsa = our_dsa.calc(self.test_data, "test", use_cache=False)

        original_dsa = np.load("./assets/original_dsa.npy")

        np.testing.assert_almost_equal(actual=test_dsa,
                                       desired=original_dsa, decimal=5)

    def test_lsa_is_consistent_with_original_implementation(self, config):

        our_lsa = LSA(model=self.model, train_data=self.train_data, config=config)
        our_lsa.prep()
        test_lsa = our_lsa.calc(self.test_data, "test", use_cache=False)
        original_lsa = np.load("./assets/original_lsa.npy")

        np.testing.assert_almost_equal(actual=test_lsa,
                                       desired=original_lsa, decimal=5)

    def test_lsa_kdes(self, config):
        nodes = 10
        our_lsa = LSA(model=self.model, train_data=self.train_data, config=config)
        our_lsa.prep()
        test_kdes, test_rm_rows = our_lsa._calc_kdes()

        self.assertIsInstance(test_kdes, dict)
        self.assertIsInstance(test_rm_rows, list)
        self.assertEqual(len(test_kdes), nodes)
        self.assertEqual(np.array(test_rm_rows).dtype, int)

    def test_dissector_sv_match(self, config):

        diss = Dissector(model=self.model, config=config)

        test_activations = np.array(([[0.7, 0.2, 0.1], [0.95, 0.04, 0.01], [0.65, 0.3, 0.05]]))
        scores = np.empty(shape = (test_activations.shape[0], ))
        true_scores = np.array([0.77777, 0.959595, 0.68421])
        for i, p in enumerate(test_activations):
            sv = diss._calc_sv_for_match(test_activations, i)
            self.assertIsInstance(sv, float)
            scores[i] = sv

        np.testing.assert_almost_equal(true_scores, scores, decimal=5)

    def test_dissector_sv_non_match(self, config):

        diss = Dissector(model=self.model, config=config)

        test_activations = np.array(([[0.7, 0.2, 0.1], [0.95, 0.04, 0.01], [0.65, 0.3, 0.05]]))
        test_p = np.array([0, 1, 1])
        scores = np.empty(shape=(test_activations.shape[0],))
        true_scores = np.array([0.12500, 0.040404, 0.31578])
        for i, p in enumerate(test_activations):
            sv = diss._calc_sv_for_non_match(test_activations, i, test_p)
            self.assertIsInstance(sv, float)
            scores[i] = sv

        np.testing.assert_almost_equal(true_scores, scores, decimal=5)

    def test_pv_scores(self, config):

        diss = Dissector(model=self.model, config=config)

        scores = np.array(([[0.1258, 0.0404, 0.3157], [0.4212, 0.3535, 0.1268], [0.3574, 0.2424, 0.3662]]))
        weights = np.array([2, 3, 5])
        true_pv_scores = np.array([0.19513, 0.25429, 0.32730])
        pv = diss.pv_scores(weights=weights, scores=scores)

        np.testing.assert_almost_equal(true_pv_scores, pv, decimal=5)








if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--implementation",
        "-implementation",
        help="SA Implementation Type [fast_sa or benchmark]",
        type=str,
        default="fast_dsa",
    )
    config = parser.parse_args()
    test_obj = TestSurpriseAdequacyConsistency()


