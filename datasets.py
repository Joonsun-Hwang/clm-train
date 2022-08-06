#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from ast import literal_eval

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HanyuaDialogDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        assert split in ['train', 'val', 'test'],\
        "[!] The argument 'split' should be 'train', 'val', or 'test"

        self.args = args
        self.split = split

        self.tokenizer = tokenizer
        with open(
                os.path.join(args.data_root_dir, args.data_dir,
                             '.'.join([split, 'jsonl']))) as i:
            self.raw_data = list(map(str.strip, i.readlines()))
            self.raw_data = list(map(literal_eval, self.raw_data))
        
        self.data = list(map(self._make_item, self.raw_data))


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _make_item(self, dialog):
        dialog_text = ''
        for utterance in dialog:
            if utterance['speaker'] == '사용자':
                utterance['speaker'] = 'User'
            elif utterance['speaker'] == 'YUA':
                utterance['speaker'] = 'Yua'
            dialog_text += utterance['speaker'] + ': ' + utterance['text'] + '\n'
        return dialog_text