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
    model_load_time = time.time()

    # Accelerator
    args.accelerator = Accelerator(cpu=args.cpu,
                                   mixed_precision=args.mixed_precision)
    args.device = args.accelerator.device

    # Tokenizer
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

    if args.saved_model:
        model.load_state_dict(
            torch.load(os.path.join(args.data_dir, 'checkpoint',
                                    'BEST_' + args.saved_model + '.ckpt'),
                    map_location=args.device))

    model = args.accelerator.prepare(model)
    args.accelerator.print('Model Loading Time:', time.time() - model_load_time)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)

    # Inference
    args.accelerator.wait_for_everyone()
    args.accelerator.print('\n\n[-] Start inference the model\n')

    while(True):
        input_text = input('User: ')
        input_text = 'User: ' + input_text + '\nAI:'
        inference_time = time.time()

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(args.device)
        outputs = model.generate(input_ids, num_beams=5, no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids, do_sample=True, top_k=50, no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids, do_sample=True, top_k=0, top_p=0.9, no_repeat_ngram_size=2)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
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
                        default='/home/jsunhwang/huggingface_models')
    parser.add_argument('--max_len', type=int, default=2048)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='skt/ko-gpt-trinity-1.2B-v0.5')
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
                                                    args.pretrained_model,
                                                    args.revision)

    logging.set_verbosity_error()

    inference(args)


if __name__ == '__main__':
    main()
