#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 




# Author: Xiaoy LI 
# Modify: Dian CHEN
# Last update: 2021.04.23 
# First create: 2019.04.01 
# Description:
# dataset_processor.py 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler 


import csv 
import logging 
import argparse 
import random 
import numpy as np 
from tqdm import tqdm 

from glyce.dataset_readers.bert_data_utils import * 


class MsraNERProcessor(DataProcessor):
    # processor for the MSRA data set 
    def get_train_examples(self, data_dir):
        # see base class 
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.ner"), delimiter=' '), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.ner"), delimiter=' '), "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.ner"), delimiter=' '), "dev")

    def get_labels(self):
        # see base class 
        return ["S-NS", "B-NS", "M-NS", "E-NS", "S-NR", "B-NR", "M-NR", "E-NR", \
        "S-NT", "B-NT", "M-NT", "E-NT", "O"]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        text_a, label = [], []
        for (i, line) in enumerate(lines):
            if not line:
                guid = "{}_{}".format("msra.ner", str(i))
                examples.append(InputExample(guid=guid, text_a=''.join(text_a), text_b=None, label=label))
                text_a, label = [], []
            else:
                text_a.append(line[0])
                label.append(line[1])
        else:
            guid = "{}_{}".format("msra.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=''.join(text_a), text_b=None, label=label))
        return examples 


class OntoNotesNERProcessor(DataProcessor):
    # processor for OntoNotes dataset 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        # see base class 
        # {'M-GPE', 'S-LOC', 'M-PER', 'B-LOC', 'E-PER', 'M-LOC', 'B-PER', 'B-GPE', 
        # 'S-ORG', 'M-ORG', 'B-ORG', 'S-GPE', 'O', 'E-GPE', 'E-LOC', 'S-PER', 'E-ORG'}
        # GPE, LOC, PER, ORG, O
        return ["O", "S-LOC", "B-LOC", "M-LOC", "E-LOC", \
        "S-PER", "B-PER", "M-PER", "E-PER", \
        "S-GPE", "B-GPE", "M-GPE", "E-GPE", \
        "S-ORG", "B-ORG", "M-ORG", "E-ORG"]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        for (i, line) in enumerate(lines):
            # print("check the content of line")
            if not line:
                continue
            line = line[0].split("\t")
            text_a = line[0].split(" ")
            text_b = None 
            label = line[1].split(" ")
            guid = "{}_{}".format("ontonotes.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=''.join(text_a), text_b=text_b, label=label))
        return examples 


class ResumeNERProcessor(DataProcessor):
    # processor for the Resume dataset 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        return ["O", "S-ORG", "S-NAME", "S-RACE", "B-TITLE", "M-TITLE", "E-TITLE", "B-ORG", "M-ORG", "E-ORG", "B-EDU", "M-EDU", "E-EDU", "B-LOC", "M-LOC", "E-LOC", "B-PRO", "M-PRO", "E-PRO", "B-RACE", "M-RACE", "E-RACE", "B-CONT", "M-CONT", "E-CONT", "B-NAME", "M-NAME", "E-NAME", ]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        for (i, line) in enumerate(lines):
            # print("check the content of line")
            if not line:
                continue
            line = line[0].split("\t")
            text_a = line[0].split(" ")
            text_b = None 
            label = line[1].split(" ")
            guid = "{}_{}".format("resume.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=''.join(text_a), text_b=text_b, label=label))
        return examples 
    
    
class WeiboNERProcessor(DataProcessor):
    # processor for the Weibo dataset 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.all.bmes"), delimiter=' '), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.all.bmes"), delimiter=' '), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.all.bmes"), delimiter=' '), "test")
    
    def collect_labels(self, ner_data):
        all_tags = set()
        for d in ner_data:
            # tags = set([x.split('-')[-1] for x in d.label])
            tags = set([x for x in d.label])
            all_tags = all_tags.union(tags)
        return sorted(list(all_tags), key=lambda x: (x.split('-')[-1], 'BMESO'.index(x.split('-')[0])))
    
    def get_labels(self):
        return ['B-GPE.NAM', 'M-GPE.NAM', 'E-GPE.NAM', 'S-GPE.NAM',
                'B-GPE.NOM', 'M-GPE.NOM', 'E-GPE.NOM', 'S-GPE.NOM',
                'B-LOC.NAM', 'M-LOC.NAM', 'E-LOC.NAM', 'S-LOC.NAM',
                'B-LOC.NOM', 'M-LOC.NOM', 'E-LOC.NOM', 'S-LOC.NOM',
                'O',
                'B-ORG.NAM', 'M-ORG.NAM', 'E-ORG.NAM', 'S-ORG.NAM', 
                'B-ORG.NOM', 'M-ORG.NOM', 'E-ORG.NOM', 'S-ORG.NOM',
                'B-PER.NAM', 'M-PER.NAM', 'E-PER.NAM', 'S-PER.NAM',
                'B-PER.NOM', 'M-PER.NOM', 'E-PER.NOM', 'S-PER.NOM']

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        text_a, label = [], []
        for (i, line) in enumerate(lines):
            if not line:
                guid = "{}_{}".format("weibo.ner", str(i))
                examples.append(InputExample(guid=guid, text_a=''.join(text_a), text_b=None, label=label))
                text_a, label = [], []
            else:
                text_a.append(line[0])
                label.append(line[1])
        else:
            guid = "{}_{}".format("weibo.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=''.join(text_a), text_b=None, label=label))
        return examples 
    
