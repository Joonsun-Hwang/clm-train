#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from accelerate import DistributedType
from datasets import load_dataset
from torch.utils.data import DataLoader


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
                                    })
        # If you change the function in dataset.map() after you run the previous code once, you should remove cache
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
        max_length = 128 if self.args.accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if self.args.accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif self.args.accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return self.tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

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


class InstructDataset():

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
                                    })
        # If you change the function in dataset.map() after you run the previous code once, you should remove cache
        self.tokenized_dataset = self.dataset.map(
            self._tokenize_function,
            remove_columns=['instruction', 'input', 'output'])

    def _make_prompt(self, examples):
        prompt = ((
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{examples["instruction"]}

### Input:
{examples["input"]}

### Response:
"""
        ) if examples["input"] else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{examples["instruction"]}

### Response:
"""))
        return prompt

    # Reference for alpaca-like data format: https://github.com/tatsu-lab/stanford_alpaca#data-release
    # Reference for vicuna-like data format: https://huggingface.co/datasets/junelee/wizard_vicuna_70k
    def _tokenize_function(self, examples):
        prompt = self._make_prompt(examples)
        len_prompt = len(
            self.tokenizer(prompt,
                           truncation=True,
                           max_length=self.args.max_len + 1)["input_ids"]) - 1

        input_ids = self.tokenizer(prompt + examples["output"],
                                   truncation=True,
                                   max_length=self.args.max_len +
                                   1)["input_ids"][:-1]  # no eos token
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len_prompt + input_ids[len_prompt:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def _collate_fn(self, examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = self.args.max_len if self.args.accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if self.args.accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif self.args.accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        # Reference: https://huggingface.co/transformers/v4.8.0/_modules/transformers/data/data_collator.html
        # print(type(examples), examples)
        # exit()

        # Make labels
        labels = [example["labels"] for example in examples
                  ] if "labels" in examples[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + pad_to_multiple_of - 1) //
                    pad_to_multiple_of * pad_to_multiple_of)

            padding_side = self.tokenizer.padding_side
            for example in examples:
                remainder = [self.tokenizer.pad_token_id
                             ] * (max_label_length - len(example["labels"]))
                if isinstance(example["labels"], list):
                    example["labels"] = (example["labels"] +
                                         remainder if padding_side == "right"
                                         else remainder + example["labels"])
                elif padding_side == "right":
                    example["labels"] = np.concatenate(
                        [example["labels"], remainder]).astype(np.int64)
                else:
                    example["labels"] = np.concatenate(
                        [remainder, example["labels"]]).astype(np.int64)

        # Make batch
        batch = self.tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.tokenizer.pad_token_id is not None:
            batch['labels'][batch['labels'] ==
                            self.tokenizer.pad_token_id] = -100

        return batch

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
