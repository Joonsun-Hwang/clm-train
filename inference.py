#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time

import evaluate
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

import accelerate
from accelerate import Accelerator


def inference(args):
    model_load_time = time.time()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        num_added_toks = tokenizer.add_special_tokens(args.special_tokens_dict)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        # torch_dtype='auto',  # There is a bug for 'facebook/opt'
        low_cpu_mem_usage=True,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        model.resize_token_embeddings(len(tokenizer))

    if args.add_adapter and args.saved_model:
        args.adapter_name = model.load_adapter(
            os.path.join('checkpoint', 'BEST_adapter_' + args.saved_model))
        model.set_active_adapters(args.adapter_name)
        args.accelerator.print('[!] Saved checkpoint is loaded')
    elif args.saved_model:
        model.load_state_dict(
            torch.load(
                os.path.join('checkpoint',
                             'BEST_' + args.saved_model + '.ckpt')))
        args.accelerator.print('[!] Saved checkpoint is loaded')
    elif args.add_adapter:
        model.add_adapter('adapter_' + args.saved_model)
        model.set_active_adapters(args.adapter_name)

    model = args.accelerator.prepare(model)
    args.accelerator.print('Model Loading Time:',
                           time.time() - model_load_time)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)

    # Inference
    args.accelerator.wait_for_everyone()
    args.accelerator.print('\n\n[-] Start inference the model\n')

    while (True):
        input_text = input('User: ')
        input_text = 'User: ' + input_text + '\nAI:'
        inference_time = time.time()

        input_ids = tokenizer.encode(input_text,
                                     return_tensors='pt').to(args.device)
        outputs = model.generate(input_ids,
                                 num_beams=5,
                                 no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids, do_sample=True, top_k=50, no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids, do_sample=True, top_k=0, top_p=0.9, no_repeat_ngram_size=2)
        output_text = tokenizer.batch_decode(outputs,
                                             skip_special_tokens=True)[0]
        output_text = output_text.replace(input_text, '', 1).strip()

        print('AI:', output_text)
        args.accelerator.print('Inference Time:', time.time() - inference_time)


def main():
    # python test.py
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--cache_root_dir',
                        type=str,
                        default='huggingface')
    parser.add_argument('--max_len', type=int, default=2048)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='EleutherAI/polyglot-ko-1.3b')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--saved_model', type=str, default=None)

    # Multi-process Parameters
    parser.add_argument('--mixed_precision',
                        type=str,
                        default='fp16',
                        choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    os.environ['TRANSFORMERS_CACHE'] = os.path.join(args.cache_root_dir,
                                                    'transformers',
                                                    args.pretrained_model,
                                                    args.revision)
    os.environ['HF_DATASETS_CACHE'] = os.path.join(args.cache_root_dir,
                                                   'datasets')
    os.environ['HF_EVALUATE_CACHE'] = os.path.join(args.cache_root_dir,
                                                   'evaluate')
    os.environ['HF_METRICS_CACHE'] = os.path.join(args.cache_root_dir,
                                                  'metrics')
    os.environ['HF_MODULES_CACHE'] = os.path.join(args.cache_root_dir,
                                                  'modules')

    # Accelerator
    args.accelerator = Accelerator(cpu=args.cpu,
                                   mixed_precision=args.mixed_precision)
    args.device = args.accelerator.device

    # Additional special tokens
    args.special_tokens_dict = {'additional_special_tokens': ['User:', 'AI:']}

    logging.set_verbosity_error()

    inference(args)


if __name__ == '__main__':
    main()
