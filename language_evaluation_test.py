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
        evaluator = language_evaluation.CocoEvaluator()
        results = evaluator.run_evaluation(SAMPLE_PREDICTIONS, SAMPLE_ANSWERS)
        pprint(results)

    def test_rouge(self):
        evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=5)
        sample_predictions = SAMPLE_PREDICTIONS * 5000
        sample_answers = SAMPLE_ANSWERS * 5000
        results = evaluator.run_evaluation(sample_predictions, sample_answers)
        #results = evaluator.run_evaluation(SAMPLE_PREDICTIONS, SAMPLE_ANSWERS)
        pprint(results)

    def test_rouge155(self):
        evaluator = language_evaluation.Rouge155Evaluator(num_parallel_calls=5)
        sample_predictions = SAMPLE_PREDICTIONS * 5000
        sample_answers = SAMPLE_ANSWERS * 5000
        results = evaluator.run_evaluation(sample_predictions, sample_answers)
        pprint(results)


if __name__ == '__main__':
    unittest.main()
