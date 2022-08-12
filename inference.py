#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time

import accelerate
import evaluate
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


def inference(args):
    # Accelerator
    args.accelerator = Accelerator(cpu=args.cpu,
                                   mixed_precision=args.mixed_precision)
    args.device = args.accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        # torch_dtype='auto',  # There is a bug for 'facebook/opt'
        low_cpu_mem_usage=True,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])
    model.load_state_dict(
        torch.load(os.path.join(args.data_root_dir, 'checkpoint',
                                'BEST_' + args.checkpoint + '.ckpt'),
                   map_location=args.device))

    model = args.accelerator.prepare(model)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)

    # Inference
    args.accelerator.wait_for_everyone()
    args.accelerator.print('\n\n[-] Start inference the model\n')

    while(True):
        break


def main():
    # python test.py
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_root_dir', type=str, default='data')
    parser.add_argument('--data_dir',
                        type=str,
                        default='hanyua_augmented_dialog')
    parser.add_argument('--cache_root_dir',
                        type=str,
                        default='/home/jsunhwang/huggingface_models')
    parser.add_argument('--max_len', type=int, default=2048)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='kakaobrain/kogpt')
    parser.add_argument('--revision', type=str, default='KoGPT6B-ryan1.5b')
    parser.add_argument('checkpoint', type=str)

    # Multi-process Parameters
    parser.add_argument('--mixed_precision',
                        type=str,
                        default='no',
                        choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    os.environ['TRANSFORMERS_CACHE'] = os.path.join(args.cache_root_dir,
                                                    args.pretrained_model,
                                                    args.revision)

    logging.set_verbosity_error()

    inference(args)


if __name__ == '__main__':
    main()
