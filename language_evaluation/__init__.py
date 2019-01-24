import colorlog
import contextlib
import os
from subprocess import call
import abc

import numpy as np

from language_evaluation.coco_caption_py3.pycocoevalcap.eval import COCOEvalCap
from language_evaluation.coco_caption_py3.pycocotools.coco import COCO
from language_evaluation.rouge import rouge_scorer, scoring


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
    def __init__(self,
                 rouge_types=["rouge1", "rouge2", "rougeL"],
                 use_stemmer=True):
        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer

    def run_evaluation(self, predicts, answers):
        scorer = rouge_scorer.RougeScorer(self.rouge_types, self.use_stemmer)
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
