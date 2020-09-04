import csv
import glob
import json
import logging
import os
import re
import string
import numpy as np
import torch
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional, Union
from scipy.special import softmax
from collections import Counter, defaultdict
from tqdm.auto import tqdm, trange
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(asdict(self)) + "\n"

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class T5GlueDataset(Dataset):
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
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                #label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                logger.info("Training examples: %s", len(examples))
                #self.features = convert_examples_to_features(examples, tokenizer, max_length=max_seq_length, label_list=label_list)
                self.features = convert_examples_to_features(examples, tokenizer, max_length=max_seq_length)
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

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

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

class CoLAProcessor(DataProcessor):
    """Processor for the SST2 data set (T5 GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = "cola sentence: " + line[3]
            text_b = ""
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


    def _read_jsonl(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(jline) for jline in f.readlines()]

class SST2Processor(DataProcessor):
    """Processor for the SST2 data set (T5 GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = "sentiment: " + line[0]
            text_b = ""
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


    def _read_jsonl(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(jline) for jline in f.readlines()]

class CombinedProcessor(DataProcessor):
    """Processor for the Combined GLUE data set (T5 GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        for task in ["CoLA", "SST-2"]:
            task_data_dir = os.path.join(data_dir, task)
            logger.info("LOOKING AT {}".format(os.path.join(task_data_dir, "train.tsv")))
            examples.extend(self._create_examples(self._read_tsv(os.path.join(task_data_dir, "train.tsv")), "train", task))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        for task in ["CoLA", "SST-2"]:
            task_data_dir = os.path.join(data_dir, task)
            examples.extend(self._create_examples(self._read_tsv(os.path.join(task_data_dir, "dev.tsv")), "dev", task))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        for task in ["CoLA", "SST-2"]:
            task_data_dir = os.path.join(data_dir, task)
            examples.extend(self._create_examples(self._read_tsv(os.path.join(task_data_dir, "test.tsv")), "test", task))
        return examples

    def _create_examples(self, lines, set_type, task):
        """Creates examples for the training, dev and test sets."""

        if task == "CoLA":
            task_prefix = "cola: "
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = task_prefix + line[3]
                text_b = ""
                label = None if set_type == "test" else line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples
            
        if task == "SST-2":
            task_prefix = "sst: "
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = task_prefix + line[0]
                text_b = ""
                label = None if set_type == "test" else line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        print (task)
        processor = processors[task]()
        #if label_list is None:
        #    label_list = processor.get_labels()
        #    logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = T5_glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    """
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]
    """

    """
    from math import ceil
    features = []
    for batch_i in tqdm(range(ceil(len(examples)/32))):
        lower_i = batch_i*32
        upper_i = min(len(examples), (batch_i+1)*32)
        batch_encoding = tokenizer(
            [(example.text_a, example.text_b) for example in examples[lower_i:upper_i]],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for i in range(len(examples[lower_i:upper_i])):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs, label=labels[lower_i+i])
            features.append(feature)
    """        

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    batch_encoding_labels = tokenizer(
        [(example.label) for example in examples],
        max_length=1,
        padding="max_length",
        truncation=True,
    )

    #print(batch_encoding_labels)

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        labels = list(map(lambda x: int(x), batch_encoding_labels["input_ids"][i]))

        feature = InputFeatures(**inputs, label=labels)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


T5_glue_tasks_num_labels = {
    "sst-2": 2,
    "cola": 2,
    "combined": 2
}

processors = {
    "sst-2": SST2Processor,
    "cola": CoLAProcessor,
    "combined": CombinedProcessor,
}

T5_glue_output_modes = {
    "sst-2": "classification",
    "cola":  "classification",
    "combined": "classification",
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def multiclass_acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    #f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        #"f1": f1,
        #"acc_and_f1": (acc + f1) / 2,
    }

def T5_glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sst-2":
        return {"acc": simple_accuracy(labels, preds)}
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    else:
        raise KeyError(task_name)







