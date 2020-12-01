# TODO in this folder we will implement integration tests.
#   Integration tests test larger workflows than unit tests and may thus take longer to implement.
#   Integration tests are typically executed only when merging into main or selectively during development.


import unittest

import numpy as np
import tensorflow as tf

import apotoma
from apotoma.surprise_adequacy import DSA


class TestSurpriseAdequacyConsistency(unittest.TestCase):

    def setUp(self) -> None:
        self.model: tf.keras.Model = None  # TODO @Rwiddhi load original model from file
        self.train_data = None  # TODO @Rwiddhi load & preprocess
        self.test_data = None  # TODO @Rwiddhi load & preprocess
        self.train_ats = None  # TODO Load from file system

    # DO this first
    def test_train_ats_calculation_against_kims_implementation(self):
        datasplit = self.train_data[0, 100], self.train_data[1, 100]

        # TODO @ Rwiddhi call method to calculate ATS

        # HERE you'll calculate the ats on your code
        nodes = None
        sa = DSA(None)
        ats, pred = sa._calculate_ats()

        # Here you load the values from kims implementation
        kim_ats = None
        kim_pred = None

        self.assertIsInstance(ats, np.ndarray)
        self.assertEqual(ats.shape, (100, nodes))
        self.assertEqual(ats.dtype, np.float64) # TODO Hardcode the actual type we want
        np.testing.assert_almost_equal(ats, kim_ats[100], decimal=5)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred, (100,))
        self.assertEqual(pred.dtype, np.int) # TODO Hardcode the actual type we want
        np.testing.assert_equal(pred, kim_pred, decimal=5)


    def test_dsa_is_consistent_with_original_implementation(self):
        our_dsa = apotoma.surprise_adequacy.DSA(model=self.model,
                                                train_data=self.train_data,
                                                args={}  # TODO @Rwiddhi args
                                                )

        original_dsa = np.load("./assets/original_dsa.npy")

        np.testing.assert_almost_equal(actual=our_dsa,
                                       desired=original_dsa, decimal=5)

    # TODO @ Rwiddhi equivalent test for LSA
