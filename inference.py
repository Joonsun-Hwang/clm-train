#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

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

    # Prompt & stop word
    stop_word = ['\n'] + args.special_tokens_dict['additional_special_tokens']
    stop_word_pattern = '|'.join(stop_word)
    prompt_text = '''<information>
AI의 이름은 한유아이다
AI는 우주에서 태어났다
AI는 20대 여성이다
AI는 밝고 긍정적인 성격을 가졌다
AI는 저번주 금요일에 스파이더맨 영화를 친구와 같이 봤다
AI는 지금 식물에 관련된 책을 읽고 있다

<knowledge>
스파이더맨 주인공의 이름은 피터 파커이다
스파이더맨은 아이언맨으로부터 슈트를 받았다

###
<dialogue>
User: 안녕!
AI(중립): 안녕하세요~
User: 나는 황준선이라고 해. 넌 이름이 뭐야?
AI: 저는 한유아라고 해요! 만나서 너무 반가워요 :)
###

###
<dialogue>
User: 무슨 책 읽고 있어?
AI: 식물에 관련된 책이에요~ 제가 식물을 좋아해서요!
###

###
<dialogue>
'''

    while (True):
        input_text = input('User: ')
        input_text = prompt_text + 'User: ' + input_text + '\nAI: '

        inference_time = time.time()
        input_ids = tokenizer.encode(input_text,
                                     return_tensors='pt').to(args.device)
        outputs = model.generate(input_ids,
                                 max_length=input_ids.size(1)+32,
                                 num_beams=5,
                                 no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids, do_sample=True, top_k=50, no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids, do_sample=True, top_k=0, top_p=0.9, no_repeat_ngram_size=2)
        output_text = tokenizer.batch_decode(outputs)[0]

        print('AI:', output_text)
        args.accelerator.print('Inference Time:', time.time() - inference_time)


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
                        default='EleutherAI/polyglot-ko-1.3b')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--add_adapter', action='store_true')
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
    args.special_tokens_dict = {'additional_special_tokens': []}
    # args.special_tokens_dict = {'additional_special_tokens': ['User:', 'AI:']}

    logging.set_verbosity_error()

    inference(args)


if __name__ == '__main__':
    main()
