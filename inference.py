#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time

import deepspeed
import torch
from accelerate import Accelerator, DistributedType
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          logging, pipeline, set_seed)

from utils import load_best_checkpoint

# from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


def inference(args):
    pipeline_load_time = time.time()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                              revision=args.revision)

    # Additional special tokens
    args.special_tokens_dict = {'additional_special_tokens': []}
    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        num_added_toks = tokenizer.add_special_tokens(args.special_tokens_dict)

    args.config = AutoConfig.from_pretrained(args.pretrained_model,
                                             revision=args.revision)

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                 revision=args.revision)
    if args.saved_model:
        _, model, tokenizer = load_best_checkpoint(args, args.saved_model)
        print('[*] Saved checkpoint is loaded')

    # Prepare pipeline
    generator = pipeline('text-generation',
                         tokenizer=tokenizer,
                         model=model,
                         device=args.local_rank)
    if args.accelerator.state.distributed_type == DistributedType.DEEPSPEED:
        generator.model = deepspeed.init_inference(
            generator.model,
            mp_size=args.world_size,
            dtype=torch.float,
            replace_method='auto',
            replace_with_kernel_inject=True)

    print('Pipeline Loading Time:', time.time() - pipeline_load_time)

    print('\n\n[-] Arguments:\n')
    print(args)

    # Inference
    print('\n\n[-] Start inference the model\n')

    # Prompt & stop word
    stop_word = ['\n'] + args.special_tokens_dict['additional_special_tokens']
    stop_word_pattern = '|'.join(stop_word)
    init_prompt = ''''''

    while True:
        user_input = input('prompt: ')
        input_text = init_prompt + user_input

        # Forward and Generation
        inference_time = time.time()
        output_texts = generator(input_text,
                                 max_new_tokens=args.max_new_tokens,
                                 num_beams=5,
                                 no_repeat_ngram_size=2,
                                 remove_invalid_values=True)
        beam_search_inference_time = time.time() - inference_time
        print('Beam Search Inference Time:', beam_search_inference_time)

        inference_time = time.time()
        output_texts += generator(input_text,
                                  max_new_tokens=args.max_new_tokens,
                                  do_sample=True,
                                  top_k=50,
                                  top_p=0.95,
                                  no_repeat_ngram_size=2,
                                  num_return_sequences=5,
                                  remove_invalid_values=True)
        nucleus_sampling_inference_time = time.time() - inference_time
        print('Nucleus Sampling Inference Time:',
              nucleus_sampling_inference_time)

        output_texts = [
            output_text['generated_text'] for output_text in output_texts
        ]

        # Post-process outputs
        for idx in range(len(output_texts)):
            output_texts[idx] = output_texts[idx].split('<|endoftext|>')[0]
            if '.' in output_texts[idx]:
                output_texts[idx] = '.'.join(
                    output_texts[idx].split('.')[:-1]) + '.'

        print()
        for idx, output_text in enumerate(output_texts):
            print('Candidate ' + str(idx))
            print(output_text)
            print()


def main():
    # python test.py
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--max_new_tokens', type=int, default=128)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='EleutherAI/polyglot-ko-3.8b')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--saved_model', type=str, default=None)

    # Multi-process Parameters
    parser.add_argument(
        '--mixed_precision',
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()

    # Accelerator
    args.accelerator = Accelerator(cpu=args.cpu,
                                   mixed_precision=args.mixed_precision)

    # Reproduciblity
    set_seed(args.random_seed)

    # Deepspeed distributed setup
    if args.accelerator.state.distributed_type == DistributedType.DEEPSPEED:
        deepspeed.init_distributed('nccl')

    logging.set_verbosity_error()

    inference(args)


if __name__ == '__main__':
    main()
