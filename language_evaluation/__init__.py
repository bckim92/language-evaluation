import colorlog
import contextlib
import os
from subprocess import call

from language_evaluation.coco_caption_py3.pycocoevalcap.eval import COCOEvalCap
from language_evaluation.coco_caption_py3.pycocotools.coco import COCO

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


class Evaluator(object):
    def __init__(self):
        pass

    def run_evaluation(self, predicts, answers, method="coco"):
        """Wrapper function for evaluation

        Args:
            predicts: list of sentences
            answers: list of sentences. For multiple GTs, list of list of sentences.
            method: evaluation method. (e.g. "coco")

        Returns:
            Dictionary with metric name in key metric result in value
        """
        colorlog.info("Run evaluation...")
        if method == "coco":
            eval_result = self._coco_evaluation(predicts, answers)
        else:
            raise NotImplementedError()

        return eval_result

    def _coco_evaluation(self, predicts, answers):
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
            coco_eval = COCOEvalCap(coco, coco_res)
            coco_eval.evaluate()

        return coco_eval.eval
