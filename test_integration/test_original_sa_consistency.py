# TODO in this folder we will implement integration tests.
#   Integration tests test larger workflows than unit tests and may thus take longer to implement.
#   Integration tests are typically executed only when merging into main or selectively during development.


import unittest

import numpy as np
import tensorflow as tf

import apotoma


class TestSurpriseAdequacyConsistency(unittest.TestCase):

    def setUp(self) -> None:
        self.model: tf.keras.Model = None  # TODO @Rwiddhi load original model from file
        self.train_data = None  # TODO @Rwiddhi load & preprocess
        self.test_data = None  # TODO @Rwiddhi load & preprocess

    def test_dsa_is_consistent_with_original_implementation(self):
        our_dsa = apotoma.surprise_adequacy.DSA(model=self.model,
                                                train_data=self.train_data,
                                                args={}  # TODO @Rwiddhi args
                                                )

        original_dsa = np.load("./assets/original_dsa.npy")

        np.testing.assert_almost_equal(actual=our_dsa,
                                       desired=original_dsa, decimal=5)

    # TODO @ Rwiddhi equivalent test for LSA
