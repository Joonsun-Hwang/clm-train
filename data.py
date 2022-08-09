#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from ast import literal_eval

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import DistributedType


def get_dataloaders(args, split):
    assert split in ['train', 'test']

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    def _tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples['input_text'],
                            truncation=True,
                            max_length=None)
        return outputs

    def _collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if args.accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(examples,
                                 padding="max_length",
                                 max_length=args.max_len,
                                 return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Load dataset and Tokenization
    dataset = load_dataset('json',
                           data_files={
                               'train':
                               os.path.join(args.data_root_dir, 'train.jsonl'),
                               'val':
                               os.path.join(args.data_root_dir, 'val.jsonl'),
                               'test':
                               os.path.join(args.data_root_dir, 'test.jsonl')
                           })
    tokenized_dataset = dataset.map(_tokenize_function,
                                    batched=True,
                                    remove_columns=['input_text'])

    # Build DataLoader and return it
    if split == 'train':
        train_loader = DataLoader(tokenized_dataset['train'],
                                  collate_fn=_collate_fn,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(tokenized_dataset['val'],
                                collate_fn=_collate_fn,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True)
        return train_loader, val_loader
    else:
        test_loader = DataLoader(tokenized_dataset['test'],
                                 collate_fn=_collate_fn,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 pin_memory=True)
        return test_loader
