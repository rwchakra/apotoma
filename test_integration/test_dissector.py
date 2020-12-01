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
        pass

