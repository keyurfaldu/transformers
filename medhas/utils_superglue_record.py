import csv
import glob
import json
import logging
import os
import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from scipy.special import softmax
from collections import Counter, defaultdict
import numpy as np


from tqdm.auto import tqdm, trange
from filelock import FileLock

import torch
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: The untokenized text of the first sequence (context of corresponding question).
        endings: The untokenized text of answer option.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: List[int]
    context: str
    candidate: str
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    guid: List[int]
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class SuperGlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length), task,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features, self.answers = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,)
                self.answers = processor.answers
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save((self.features, self.answers), cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]



class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_jsonl(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(jline) for jline in f.readlines()]


class RecordProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def __init__(self):
        self.answers = {}

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        self.answers["train"] = {}
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        self.answers["dev"] = {}
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        self.answers["test"] = {}
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        #if set_type == "train":
        #    lines = lines[:1000]
        #else:
        #    lines = lines[:200]

        examples = []
        for i in tqdm(range(len(lines))):
            line = lines[i]
            if i == 0:
                continue
            passage_idx = line["idx"]
            passage_text = line["passage"]["text"]
            entities = line["passage"]["entities"]
            answer_candidates = []
            for ent in entities:
                if ent["end"] > 512:
                    continue
                entity = passage_text[ent["start"]:(ent["end"]+1)]
                answer_candidates.append(entity)

            questions = line["qas"]
            for j, question in enumerate(questions):
                question_text = question['query']
                question_idx = question['idx']
                #answers = list(map(lambda x: x['text'], question['answers']))
                q_answers=[]
                for answer in question['answers']:
                    if answer['end'] > 512:
                        continue
                    q_answers.append(answer['text'])

                if len(q_answers) == 0:
                    continue
                
                self.answers[set_type][(passage_idx, question_idx)] = (answer_candidates, q_answers)

                for k, entity in enumerate(answer_candidates):
                    guid = [passage_idx, question_idx, k]
                    label = None if set_type == "test" else str(int(entity in q_answers))
                    context = passage_text
                    question = question_text.replace("@placeholder", entity)
                    examples.append(InputExample(
                        guid=guid, 
                        context=context, 
                        candidate=question, 
                        label=label))
        return examples

def convert_examples_to_features(
    examples: List[InputExample], label_list: List[str], max_length: int, tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    #for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
    for ex_index in tqdm(range(len(examples)), desc="Convert Examples to Features"):
        example = examples[ex_index]
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        inputs = tokenizer(
            example.context,
            example.candidate,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens (swag task is ok). "
                "If you are training ARC and RACE and you are poping question + options,"
                "you need to try to use a bigger max seq length!"
            )
    
        features.append(
            InputFeatures(
                guid=example.guid,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                label=label_map[example.label],
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


processors = { 
    "record": RecordProcessor,
    }

superglue_tasks_num_labels = {
    "record": 2,
    }

superglue_output_modes = {
    "record": "classification",
}

def simple_accuracy(preds, labels, guids):
    return (preds == labels).mean()

def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script """
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """ Compute max metric between prediction and each ground truth.
    From official ReCoRD eval script """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def _record_f1_score(prediction, ground_truth):
    """ Compute normalized token level F1
    From official ReCoRD eval script """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def _record_em_score(prediction, ground_truth):
    """ Compute normalized exact match
    From official ReCoRD eval script """
    return normalize_answer(prediction) == normalize_answer(ground_truth)



def superglue_compute_metrics(task_name, preds, labels, guids=None, answers=None):
    assert len(preds) == len(labels)
    if task_name == "record":
        assert len(guids) == len(preds), "Different number of predictions and IDs!"
        qst2ans = defaultdict(list)
        # iterate over examples and aggregate statistics
        for idx, pred, label in zip(guids, preds, labels):
            qst_idx = (idx[0], idx[1])
            qst2ans[qst_idx].append((idx[2], pred))
        f1s, ems = [], []
        for qst, idxs_and_prds in qst2ans.items():
            cands, golds = answers[qst]
            idxs_and_prds.sort(key=lambda x: x[0])
            logits = np.vstack([i[1] for i in idxs_and_prds])
            # take the most probable choice as the prediction
            pred_idx = softmax(logits, axis=1)[:, -1].argmax().item()
            pred = cands[pred_idx]
            # compute metrics
            f1 = metric_max_over_ground_truths(_record_f1_score, pred, golds)
            em = metric_max_over_ground_truths(_record_em_score, pred, golds)
            f1s.append(f1)
            ems.append(em)
        avg_f1 = sum(f1s) / len(f1s)
        avg_em = sum(ems) / len(ems)
        em_and_f1 = (avg_em + avg_f1) / 2
        return {"f1": avg_f1, "em": avg_em, "em_and_f1": em_and_f1}


    else:
        raise KeyError(task_name)