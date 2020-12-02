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


class TestSurpriseAdequacyConsistency(unittest.TestCase):

    def setUp(self) -> None:
        # TODO @Rwiddhi. Move the h5 file to the assets folder
        #   and use a relative path to refer to it (tests/asset/model_mnist.h5)
        self.model: tf.keras.Model = load_model(
            '/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/model/model_mnist.h5')
        (self.train_data, _), (self.test_data, y_test) = mnist.load_data()
        self.train_data = self.train_data.reshape(-1, 28, 28, 1)
        self.test_data = self.test_data.reshape(-1, 28, 28, 1)

        self.train_data = self.train_data.astype("float32")
        self.train_data = (self.train_data / 255.0) - (1.0 - 0.5)
        self.test_data = self.test_data.astype("float32")
        self.test_data = (self.test_data / 255.0) - (1.0 - 0.5)

    # DO this first
    def test_train_ats_calculation_against_kims_implementation(self):
        datasplit_train, datasplit_test = self.train_data[0:100], self.test_data[0:100]

        # TODO @Rwiddhi. Create instance of new config class instead of args
        args = {'d': 'mnist', 'is_classification': True,
                'dsa': True, 'lsa': False, 'batch_size': 128,
                'var_threshold': 1e-5, 'upper_bound': 2000,
                'n_bucket': 1000, 'num_classes': 10,
                'layer_names': ['activation_3'], 'saved_path': './tmp1/'}

        # HERE you'll calculate the ats on your code
        nodes = 10
        sa = DSA(self.model, datasplit_train, args)
        ats, pred = sa._calculate_ats()

        # Here you load the values from kims implementation
        kim_ats = np.load('/tests/assets/mnist_train_activation_3_ats.npy')
        # TODO @Rwiddhi. Move the file to the assets folder
        #   and use a relative path to refer to it (tests/asset/...)
        kim_pred = np.load('/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/tmp/mnist_train_pred.npy.npy')

        self.assertIsInstance(ats, np.ndarray)
        self.assertEqual(ats.shape, (100, nodes))
        self.assertEqual(ats.dtype, np.float32)
        np.testing.assert_almost_equal(ats, kim_ats[0:100], decimal=5)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred, (100,))
        self.assertEqual(pred.dtype, np.int)
        np.testing.assert_equal(pred, kim_pred)

    def test_dsa_is_consistent_with_original_implementation(self):
        # TODO @Rwiddhi. Create instance of new config class instead of args
        args = {'d': 'mnist', 'is_classification': True,
                'dsa': True, 'lsa': False, 'batch_size': 128,
                'var_threshold': 1e-5, 'upper_bound': 2000,
                'n_bucket': 1000, 'num_classes': 10,
                'layer_names': ['activation_3'], 'saved_path': './tmp1/'}

        our_dsa = DSA(model=self.model, train_data=self.train_data, args=args)
        our_dsa.prep()
        test_dsa = our_dsa.calc(self.test_data, "test", use_cache=False)

        original_dsa = np.load("./assets/original_dsa.npy")

        np.testing.assert_almost_equal(actual=test_dsa,
                                       desired=original_dsa, decimal=5)

    def test_lsa_is_consistent_with_original_implementation(self):
        # TODO @Rwiddhi. Create instance of new config class instead of args
        args = {'d': 'mnist', 'is_classification': True,
                'dsa': True, 'lsa': False, 'batch_size': 128,
                'var_threshold': 1e-5, 'upper_bound': 2000,
                'n_bucket': 1000, 'num_classes': 10,
                'layer_names': ['activation_3'], 'saved_path': './tmp1/'}

        our_lsa = LSA(model=self.model, train_data=self.train_data, args=args)
        our_lsa.prep()
        test_lsa = our_lsa.calc(self.test_data, "test", use_cache=False)
        original_lsa = np.load("./assets/original_lsa.npy")

        np.testing.assert_almost_equal(actual=test_lsa,
                                       desired=original_lsa, decimal=5)

    def test_lsa_kdes(self):
        nodes = 10
        # TODO @Rwiddhi. Create instance of new config class instead of args
        args = {'d': 'mnist', 'is_classification': True,
                'dsa': True, 'lsa': False, 'batch_size': 128,
                'var_threshold': 1e-5, 'upper_bound': 2000,
                'n_bucket': 1000, 'num_classes': 10,
                'layer_names': ['activation_3'], 'saved_path': './tmp1/'}

        our_lsa = LSA(model=self.model, train_data=self.train_data, args=args)
        our_lsa.prep()
        test_kdes, test_rm_rows = our_lsa._calc_kdes()

        self.assertIsInstance(test_kdes, dict)
        self.assertIsInstance(test_rm_rows, list)
        self.assertEqual(len(test_kdes), nodes)
        self.assertEqual(np.array(test_rm_rows).dtype, int)
