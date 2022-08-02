import os
import json

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

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
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
        return dialog