'''Unit Test for language_evaluation'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from pprint import PrettyPrinter
import sys

import language_evaluation

pprint = PrettyPrinter().pprint
SAMPLE_PREDICTIONS = ['i am a boy', 'she is a girl']
SAMPLE_ANSWERS = ['am i a boy ?', 'is she a girl ?']

class TestExample(unittest.TestCase):
    """ Basic uint test.  """

    def setUp(self):
        sys.stdout.write('\n')

    # ------------------------------------------------------------------------
    # Basic functionality tests
    def test_coco(self):
        evaluator = language_evaluation.Evaluator()
        results = evaluator.run_evaluation(
            SAMPLE_PREDICTIONS, SAMPLE_ANSWERS, method="coco")
        pprint(results)

    def test_rouge(self):
        evaluator = language_evaluation.Evaluator()
        results = evaluator.run_evaluation(
            SAMPLE_PREDICTIONS, SAMPLE_ANSWERS, method="rouge")
        pprint(results)


if __name__ == '__main__':
    unittest.main()
