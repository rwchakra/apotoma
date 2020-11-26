# TODO in this folder we will implement a small test for *every* method.
#   these tests should test the methods as *units*, i.e., in isolation.
#   Thus, they should be fast to run on any machine.


import unittest

import apotoma


class TestSurpriseAdequacyConsistency(unittest.TestCase):

    def test_dummy(self):
        """
        This is just a dummy unit test.
        """
        lsa_class = apotoma.surprise_adequacy.LSA
        self.assertIsNotNone(lsa_class)
