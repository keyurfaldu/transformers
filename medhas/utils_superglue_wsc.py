import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from tqdm.auto import tqdm, trange
from tqdm.auto import tqdm, trange
from filelock import FileLock
from itertools import compress 


import torch
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpanClassificationExample:
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

    guid: str
    text_a: str
    span_a: Tuple[int]
    span_b: Tuple[int]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    guid: str
    input_ids: List[int]
    attention_mask: Optional[List[int]]
    token_type_ids: Optional[List[int]]
    spans: [List[Tuple[int]]]
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
                self.features = torch.load(cached_features_file)
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
                self.examples = examples
                self.features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,)
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

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


class WSCProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def __init__(self):
        self.answers = {}

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [False, True]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["text"]
            span_a = (line["target"]["span1_index"], line["target"]["span1_index"] + len(line["target"]["span1_text"]))
            span_b = (line["target"]["span2_index"], line["target"]["span2_index"] + len(line["target"]["span2_text"]))
            label = line["label"] if "label" in line else False
            examples.append(
                SpanClassificationExample(
                    guid=guid, text_a=text_a, span_a=span_a, span_b=span_b, label=label
                )
            )
        return examples
       

def tokenize_tracking_span(tokenizer, text, span):
    toks = tokenizer.encode_plus(text, return_token_type_ids=True)
    full_toks = toks["input_ids"]
    prefix_len = len(tokenizer.decode(full_toks[:1])) + 1  # add a space
    len_covers = []
    for i in range(2, len(full_toks)):
        partial_txt_len = len(tokenizer.decode(full_toks[:i], clean_up_tokenization_spaces=False))
        len_covers.append(partial_txt_len - prefix_len)

    start, end = span
    start_tok, end_tok = None, None
    for tok_n, len_cover in enumerate(len_covers):
        if len_cover >= start and start_tok is None:
            start_tok = tok_n + 1  # account for [CLS] tok
        if len_cover >= end:
            assert start_tok is not None
            end_tok = tok_n + 1
            break
    assert start_tok is not None, "start_tok is None!"
    assert end_tok is not None, "end_tok is None!"
    span_locs = (start_tok, end_tok)
    return toks, span_locs

def convert_examples_to_features(
    examples: List[SpanClassificationExample], label_list: List[str], max_length: int, tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    #for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
    print(len(examples))
    for ex_index in tqdm(range(len(examples)), desc="Convert Examples to Features"):
        example = examples[ex_index]
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        inputs = tokenizer(
            example.text_a,
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
    
        _, span_locs_a = tokenize_tracking_span(tokenizer, example.text_a, example.span_a)
        _, span_locs_b = tokenize_tracking_span(tokenizer, example.text_a, example.span_b)
        #span_mask=[0]*len(inputs["input_ids"])
        #span_mask[span_locs_a[0]:span_locs_a[1]+1] = [1]*(span_locs_a[1]+1-span_locs_a[0])
        #span_mask[span_locs_b[0]:span_locs_b[1]+1] = [1]*(span_locs_b[1]+1-span_locs_b[0])

        #spans_selected = tokenizer.convert_ids_to_tokens(list(compress(inputs["input_ids"], span_mask)))
        #if(ex_index % 100 == 0):
        #    print (spans_selected)

        features.append(
            InputFeatures(
                guid=example.guid,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                spans=[span_locs_a, span_locs_b],
                label=label_map[example.label],
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


processors = { 
    "wsc": WSCProcessor,
    }

superglue_tasks_num_labels = {
    "wsc": 2,
    }

superglue_output_modes = {
    "wsc": "span-classification",
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def superglue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "wsc":
        return {"acc": simple_accuracy(labels, preds)}
    else:
        raise KeyError(task_name)