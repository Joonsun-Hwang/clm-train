#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time

import deepspeed
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          logging, pipeline)

from accelerate import Accelerator
from model import GPTNeoXPrefixForCausalLM
from utils import str2bool


def inference(args):
    pipeline_load_time = time.time()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Additional special tokens
    args.special_tokens_dict = {'additional_special_tokens': []}
    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        num_added_toks = tokenizer.add_special_tokens(args.special_tokens_dict)

    args.config = AutoConfig.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Model
    if args.p_tuning:
        args.config.pre_seq_len = args.pre_seq_len
        args.config.prefix_projection = args.prefix_projection
        args.config.prefix_hidden_size = args.prefix_hidden_size
        args.config.hidden_dropout_prob = args.hidden_dropout_prob
        model = GPTNeoXPrefixForCausalLM.from_pretrained(
            args.pretrained_model,
            revision=args.revision,
            config=args.config,
            cache_dir=os.environ['TRANSFORMERS_CACHE'])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model,
            revision=args.revision,
            cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if args.add_adapter:
        assert args.saved_model
        args.adapter_name = model.load_adapter(
            os.path.join('checkpoint', 'BEST_adapter_' + args.saved_model))
        model.set_active_adapters(args.adapter_name)
        args.accelerator.print('[!] Saved checkpoint is loaded')
    elif args.saved_model:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join('checkpoint', 'BEST_' + args.saved_model))
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join('checkpoint', 'BEST_' + args.saved_model))
        args.accelerator.print('[!] Saved checkpoint is loaded')

    # Prepare pipeline
    generator = pipeline('text-generation',
                         tokenizer=tokenizer,
                         model=model,
                         device=args.local_rank)
    # generator.model = deepspeed.init_inference(generator.model,
    #                                            mp_size=1,
    #                                            dtype=torch.float,
    #                                            replace_method='auto',
    #                                            replace_with_kernel_inject=True)

    args.accelerator.print('Pipeline Loading Time:',
                           time.time() - pipeline_load_time)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)

    # Inference
    args.accelerator.print('\n\n[-] Start inference the model\n')

    # Prompt & stop word
    stop_words = ['\n', '.', '!', '?'
                  ] + args.special_tokens_dict['additional_special_tokens']
    stop_word_pattern = '|'.join(stop_words)
    init_prompt = '''<|Information|>
나의 이름은 황준선이다.
29살 남자다.
나는 인공지능 연구원이다.
나는 지금 회사에서 야근하고 있다.
나는 이번 주말에 로스트아크 게임을 할 것이다.

<|Dialogue|>:
<|User|>: 안녕? 나는 김석겸이라고 해. 뭐하고 있어?
<|AI|>: 안녕하세요. 저는 황준선이라고 합니다. 회사에서 야근 중이에요.
'''

    while True:
        user_utterance = input('<|User|>: ')
        input_text = init_prompt + '<|User|>: ' + user_utterance + '\n<|AI|>: '

        inference_time = time.time()
        output_texts = generator(input_text,
                                 max_new_tokens=32,
                                 num_beams=5,
                                 no_repeat_ngram_size=2,
                                 remove_invalid_values=True)

        output_texts = [
            output_text['generated_text'] for output_text in output_texts
        ]

        # Post-process outputs
        for idx in range(len(output_texts)):
            output_texts[idx] = output_texts[idx].split(
                user_utterance)[-1].strip().split('\n')[0].replace(
                    '<|AI|>', '').replace(':', '').strip()
            output_text_tmp = output_texts[idx]
            for stop_word in stop_words:
                output_text_tmp = output_text_tmp.split(stop_word)[-1]
            output_texts[idx] = output_texts[idx].replace(output_text_tmp, '')

        args.accelerator.print('<|AI|>:', output_texts[0])
        args.accelerator.print('Inference Time:', time.time() - inference_time)

        init_prompt += '<|User|>: ' + input_text + '\n<|AI|>: ' + output_texts[
            0] + '\n'


def main():
    # python test.py
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--cache_root_dir', type=str, default='huggingface')
    parser.add_argument('--max_len', type=int, default=2048)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='EleutherAI/polyglot-ko-3.8b')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--add_adapter', action='store_true')
    parser.add_argument('--p_tuning', action='store_true')
    parser.add_argument('--saved_model', type=str, default=None)

    # Tuning Parameters
    parser.add_argument('--pre_seq_len', type=int, default=10)
    parser.add_argument('--prefix_projection', type=str2bool, default=True)
    parser.add_argument('--prefix_hidden_size', type=int, default=512)
    parser.add_argument('--hidden_dropout_prob', type=float, default=.1)

    # Multi-process Parameters
    parser.add_argument('--mixed_precision',
                        type=str,
                        default='fp16',
                        choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

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

    logging.set_verbosity_error()

    inference(args)


if __name__ == '__main__':
    main()
