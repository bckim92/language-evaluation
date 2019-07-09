from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import colorlog
import contextlib
import os
from subprocess import call
import abc
import shutil
from tempfile import mkdtemp
import json

import numpy as np
import more_itertools

from language_evaluation.coco_caption_py3.pycocoevalcap.eval import COCOEvalCap
from language_evaluation.coco_caption_py3.pycocotools.coco import COCO
from language_evaluation.rouge import rouge_scorer, scoring
from language_evaluation.pyrouge.Rouge155 import Rouge155


__PATH__ = os.path.abspath(os.path.dirname(__file__))


def download(name):
    if name == "coco":
        _download_coco()
    else:
        raise NotImplementedError()


def _download_coco():
    COCO_PATH = os.path.join(__PATH__, "coco_caption_py3")

    # Download models for SPICE
    SPICE_PATH = os.path.join(COCO_PATH, "pycocoevalcap/spice/lib")
    CORENLP_DATE = "stanford-corenlp-full-2015-12-09"
    CORENLP_VER = "stanford-corenlp-3.6.0"
    CORENLP_URL = "http://nlp.stanford.edu/software/{}.zip".format(CORENLP_DATE)
    CORENLP_ZIP = os.path.join(__PATH__, "{}.zip".format(CORENLP_DATE))

    CORENLP_JAR = os.path.join(SPICE_PATH, CORENLP_DATE, f"{CORENLP_VER}.jar")
    CORENLP_MODELS_JAR = os.path.join(SPICE_PATH, CORENLP_DATE, f"{CORENLP_VER}-models.jar")

    if not os.path.exists(os.path.join(SPICE_PATH, "stanford-corenlp-3.6.0.jar")):
        call(f"wget -O {CORENLP_ZIP} {CORENLP_URL}".split())
        call(f"unzip {CORENLP_ZIP} -d {SPICE_PATH}/".split())
        call(f"mv {CORENLP_JAR} {SPICE_PATH}/".split())
        call(f"mv {CORENLP_MODELS_JAR} {SPICE_PATH}/".split())
        call(f"rm -f {CORENLP_ZIP}".split())
        call(f"rm -rf {SPICE_PATH}/{CORENLP_VER}/".split())
    else:
        print(f"${SPICE_PATH}/{CORENLP_VER}.jar already exists")

    # Download models for METEOR
    PARAPHRASE_EN_URL="https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase-en.gz"
    METEOR_DATA_FNAME=os.path.join(COCO_PATH, "pycocoevalcap/meteor/data/paraphrase-en.gz")
    if not os.path.exists(METEOR_DATA_FNAME):
        call(f"mkdir -p {COCO_PATH}/pycocoevalcap/meteor/data".split())
        call(f"wget -O {METEOR_DATA_FNAME} {PARAPHRASE_EN_URL}".split())
    else:
        print(f"{METEOR_DATA_FNAME} already exists")


def _period_sentence_splitter(text: str) -> List[str]:
    return text.split(".")


def _split_list(in_list, num_splits):
    return [list(c) for c in more_itertools.divide(num_splits, in_list)]


class Evaluator(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run_evaluation(self, predicts, answers):
        pass



class CocoEvaluator(Evaluator):
    def __init__(self,
                 coco_types=["BLEU", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]):
        self.coco_types = coco_types

    def run_evaluation(self, predicts, answers):

        coco_res = []
        ann = {'images': [], 'info': '', 'type': 'captions', 'annotations': [], 'licenses': ''}

        for i, (predict, _answers) in enumerate(zip(predicts, answers)):
            predict_cap = ' '.join(predict)

            if type(_answers) == str:
                _answers = [_answers]
            answer_caps = []
            for _answer in _answers:
                answer_cap = ' '.join(_answer).replace('_UNK', '_UNKNOWN')
                answer_caps.append(answer_cap)

            ann['images'].append({'id': i})
            for answer_cap in answer_caps:
                ann['annotations'].append({'caption': answer_cap, 'id': i, 'image_id': i})
            coco_res.append({'caption': predict_cap, 'id': i, 'image_id': i})

        with contextlib.redirect_stdout(None):
            coco = COCO(ann)
            coco_res = coco.loadRes(coco_res)
            coco_eval = COCOEvalCap(coco, coco_res, self.coco_types)
            coco_eval.evaluate()

        return coco_eval.eval


class RougeEvaluator(Evaluator):
    """Calculate rouges scores two blobs of single-sentence text by using
    google's python rouge scripts.
    (If you wnat to get sentence-level ROUGE-L, use Rouge155Evaluator)

    Sample usage:
        evaluator = language_evaluation.RougeEvaluator(
            rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        results = evaluator.run_evaluation(
            ['i am a boy', 'she is a girl'],
            ['am i a boy ?', 'is she a girl ?'])
    """
    def __init__(self,
                 num_parallel_calls: int = 1,
                 rouge_types=["rouge1", "rouge2", "rougeL"],
                 use_stemmer=True,
                 tokenization_fn=None):
        self._num_parallel_calls = num_parallel_calls
        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self._tokenization_fn = tokenization_fn

    def run_evaluation(self, predicts, answers):
        n_predicts = _split_list(predicts, self._num_parallel_calls)
        n_answers = _split_list(answers, self._num_parallel_calls)
        from multiprocessing import Pool
        p = Pool(self._num_parallel_calls)
        import time
        start = time.time()
        results = p.map(self._run_evaluation, zip(n_predicts, n_answers))
        p.close()
        p.join()
        end = time.time()
        print(f"Takes {end-start} seconds for rouge evaluation with \
              {self._num_parallel_calls} processes")

        # Average results form processes
        averaged_result = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for result in results:
            for key, value in result.items():
                averaged_result[key].append(value)
        for key, value in averaged_result.items():
            # TODO : Currently, we assume each process has same numver of
            # predict-answer pairs
            averaged_result[key] = sum(value) / len(value)

        return averaged_result

    def _run_evaluation(self, predicts_and_answers):
        predicts, answers = predicts_and_answers
        scorer = rouge_scorer.RougeScorer(self.rouge_types, self.use_stemmer, self._tokenization_fn)
        scores = {rouge_type: [] for rouge_type in self.rouge_types}
        for predict, answer in zip(predicts, answers):
            # TODO : support multi-reference
            score = scorer.score(answer, predict)
            for key, value in score.items():
                scores[key].append(value.fmeasure)

        # Averaging
        for key in scores.keys():
            scores[key] = np.mean(np.array(scores[key]))

        return scores


class Rouge155Evaluator(Evaluator):
    """Calculate rouges scores two blobs of multi-sentence text by using
    the original ROUGE-1.5.5 perl script.
    It takes multi-sentence text as a string and (by default) split sentences
    based on the period symbol.
    For speed up, pass `num_parallel_calls` > 2.

    Sample usage:
        evaluator = language_evaluation.Rouge155Evaluator(
            rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        results = evaluator.run_evaluation(
            ['i am a boy . she is a girl'],
            ['i am a boy . she is not a girl'])
    """
    def __init__(self,
                 num_parallel_calls: int = 1,
                 sentence_splitter=_period_sentence_splitter,
                 rouge_args="-a -c 95 -m -n 2 -w 1.2"):
        self._num_parallel_calls = num_parallel_calls
        self._sentence_splitter = sentence_splitter
        # Rouge arguments with rouge-related data path
        self._pyrouge_path = os.path.join(__PATH__, 'pyrouge', 'RELEASE-1.5.5')
        self._rouge_args = f"-e {self._pyrouge_path}/data {rouge_args}"

        # pyrouge takes input as dumpped file
        self._tmp_path = mkdtemp()
        self._dummy_empty_string = "dummystringforemptyprediction"

        # For safe html text
        # (Following https://github.com/abisee/pointer-generator/blob/master/decode.py#L201)
        self._left_angled_bracket = "&lt;"
        self._right_angled_bracket = "&gt;"

    def run_evaluation(self, predicts, answers):
        ratio_in_split = \
            self._set_output_path_and_dump_sentences(predicts, answers)

        from multiprocessing import Pool
        p = Pool(self._num_parallel_calls)
        import time
        start = time.time()
        results = p.map(self._run_pyrouge, enumerate(ratio_in_split))
        p.close()
        p.join()
        end = time.time()
        print(f"Takes {end-start} seconds for Rouge155 evaluation with \
              {self._num_parallel_calls} processes")

        # Average results form processes
        averaged_result = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for result in results:
            for key, value in result.items():
                averaged_result[key].append(value)
        for key, value in averaged_result.items():
            averaged_result[key] = sum(value)

        # Cleanup
        shutil.rmtree(self._tmp_path)

        return averaged_result

    def _set_output_path_and_dump_sentences(self, predicts, answers):
        # Set N output paths
        if os.path.exists(self._tmp_path):
            shutil.rmtree(self._tmp_path)
        os.makedirs(self._tmp_path)
        for i in range(self._num_parallel_calls):
            os.makedirs(os.path.join(self._tmp_path, str(i)))
            os.makedirs(os.path.join(self._tmp_path, str(i), 'pred'))
            os.makedirs(os.path.join(self._tmp_path, str(i), 'answer'))

        # Divide sentences into N list
        n_predicts = _split_list(predicts, self._num_parallel_calls)
        n_answers = _split_list(answers, self._num_parallel_calls)
        ratio_in_split = [len(n_answer) / len(answers) for n_answer in n_answers]

        # Dump N-divided sentences
        for n, (n_predict, n_answer) in enumerate(zip(n_predicts, n_answers)):
            for i, (predict, answer) in enumerate(zip(n_predict, n_answer)):
                predict_str = self._make_html_safe(
                    '\n'.join(self._sentence_splitter(predict)))
                answer_str = self._make_html_safe(
                    '\n'.join(self._sentence_splitter(answer)))

                if answer_str == '':
                    continue
                if predict_str == '':
                    predict_str = self._dummy_empty_string

                pred_fname = os.path.join(
                    self._tmp_path, f"{n}/pred/pred{i}.txt")
                answer_fname = os.path.join(
                    self._tmp_path, f"{n}/answer/answer{i}.txt")
                with open(pred_fname, 'w') as fp:
                    fp.write(predict_str)
                with open(answer_fname, 'w') as fp:
                    fp.write(answer_str)

        return ratio_in_split

    def _run_pyrouge(self, idx_and_ratio):
        process_idx, ratio = idx_and_ratio
        r = Rouge155(rouge_dir=self._pyrouge_path)
        r.system_dir = os.path.join(self._tmp_path, str(process_idx), 'pred')
        r.model_dir = os.path.join(self._tmp_path, str(process_idx), 'answer')
        r.system_filename_pattern = 'pred(\d+).txt'
        r.model_filename_pattern = 'answer#ID#.txt'
        rouge_results = r.convert_and_evaluate(rouge_args=self._rouge_args)
        rouge_results = r.output_to_dict(rouge_results)
        result_dict = {'rouge1': rouge_results['rouge_1_f_score'] * ratio,
                       'rouge2': rouge_results['rouge_2_f_score'] * ratio,
                       'rougeL': rouge_results['rouge_l_f_score'] * ratio}

        # Cleanup
        shutil.rmtree(r._config_dir)
        shutil.rmtree(r._output_dir)

        return result_dict

    def _make_html_safe(self, sentence):
        """Replace any angled brackets in string to avoid interfering with HTML
        """
        sentence = sentence.\
            replace("<", self._left_angled_bracket).\
            replace(">", self._right_angled_bracket)
        return sentence
