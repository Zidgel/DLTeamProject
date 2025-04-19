# Hayden Schennum
# 2025-04-18

import unittest
from AdaIN import *



class TestTheModule(unittest.TestCase):

    def test_1(self):
        self.assertEqual(get_hash_output("rn=1"),30)
        self.assertEqual(get_hash_output("cm-"),253)
        self.assertEqual(get_hash_output("qp=3"),97)