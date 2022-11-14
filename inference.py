#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time

import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, pipeline

from accelerate import Accelerator


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

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        model.resize_token_embeddings(len(tokenizer))

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
    stop_word = ['\n'] + args.special_tokens_dict['additional_special_tokens']
    stop_word_pattern = '|'.join(stop_word)
    init_prompt = '''<|title|>
달이 떴다고 전화를 주시다니요

<|lyrics|>
세상에
강변에 달빛이 곱다고
전화를 다 주시다니요
'''

    while True:
        user_input = input('prompt: ')
        input_text = init_prompt + user_input

        # Forward and Generation
        inference_time = time.time()
        output_texts = generator(input_text,
                                 max_new_tokens=args.max_len,
                                 num_beams=5,
                                 no_repeat_ngram_size=2,
                                 remove_invalid_values=True)
        beam_search_inference_time = time.time() - inference_time
        print('Beam Search Inference Time:', beam_search_inference_time)

        inference_time = time.time()
        output_texts += generator(input_text,
                                  max_new_tokens=args.max_len,
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

        args.accelerator.print()
        for idx, output_text in enumerate(output_texts):
            args.accelerator.print('Candidate ' + str(idx))
            args.accelerator.print(output_text)
            args.accelerator.print()


# examples:

# 많이 힘들었냐는 누군가의 질문에 쉽사리 대답하지 못할 때가 많아요. 길었던 수많은 밤들을 어떠한 말로도 설명할 수가 없을 것 같아서.
# 양치기 산티아고가 양떼를 데리고 버려진 낡은 교회 앞에 다다랐을 때는 날이 저물고 있었다. 지붕은 무너진지 오래였고, 성물 보관소 자리에는 커다란 무화과나무 한 그루가 서 있었다.\n그는 그곳에서 하룻밤을 보내기로 했다. 양들을 부서진 문을 통해 안으로 들여보낸 뒤, 도망치지 못하도록 문에 널빤지를 댔다. 근처에 늑대는 없었지만, 밤사이 양이 한마리라도 도망치게 되면 그 다음날은 온종일 잃어버린 양을 찾아다녀야 할 것이기 때문이었다.
# '''
# 우리는 모두 땅에서 태어났다.
# 햇살이 문득 따사로운 날들이 있다.
# 이것은 어느 긴긴 밤에 시작된 이야기다.
# '''

# poetry examples:
# <|title|>\n달이 떴다고 전화를 주시다니요\n\n<|lyrics|>\n세상에\n강변에 달빛이 곱다고\n전화를 다 주시다니요\n


def main():
    # python test.py
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--cache_root_dir', type=str, default='huggingface')
    parser.add_argument('--max_len', type=int, default=128)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='EleutherAI/polyglot-ko-3.8b')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--add_adapter', action='store_true')
    parser.add_argument('--saved_model', type=str, default=None)

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
