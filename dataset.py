#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from datasets import load_dataset
from torch.utils.data import DataLoader

from accelerate import DistributedType


class CausalDataset():

    def __init__(self, args, tokenizer):
        self.args = args

        # Tokenizer
        self.tokenizer = tokenizer

        # Load dataset
        self.dataset = load_dataset('json',
                                    data_files={
                                        'train':
                                        os.path.join(args.data_dir,
                                                     'train.jsonl'),
                                        'val':
                                        os.path.join(args.data_dir,
                                                     'val.jsonl'),
                                        'test':
                                        os.path.join(args.data_dir,
                                                     'test.jsonl')
                                    },
                                    cache_dir=os.environ['HF_DATASETS_CACHE'])
        # If you change the function in dataset.map() after you run the previous code once, you should remove cache huggingface/datasets
        self.tokenized_dataset = self.dataset.map(
            self._tokenize_function, remove_columns=['prompt', 'completion'])

    def _tokenize_function(self, examples):
        # max_length=None => use the model max length (it's actually the default)
        encoded = self.tokenizer(examples['prompt'] + ' ' +
                                 examples['completion'],
                                 truncation=True,
                                 max_length=self.args.max_len)
        encoded['labels'] = self.tokenizer.encode(' ' + examples['completion'],
                                                  truncation=True,
                                                  max_length=self.args.max_len)
        encoded['labels'] = [-100] * (len(encoded['input_ids']) - len(
            encoded['labels'])) + encoded['labels']
        return encoded

    def _collate_fn(self, examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if self.args.accelerator.distributed_type == DistributedType.TPU:
            return self.tokenizer.pad(examples,
                                      padding="max_length",
                                      max_length=self.args.max_len,
                                      return_tensors="pt")
        return self.tokenizer.pad(
            examples,
            padding="longest",
            #   padding="max_length",
            #   max_length=self.args.max_len,  # for test, check args.extra_memory
            return_tensors="pt")

    def get_dataloaders(self, split):
        assert split in ['train', 'validation', 'test']

        # Build DataLoader and return it
        if split == 'train':
            loader = DataLoader(self.tokenized_dataset['train'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)
        elif split == 'validation':
            loader = DataLoader(self.tokenized_dataset['val'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True)
        else:
            loader = DataLoader(self.tokenized_dataset['test'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True)
        return loader
