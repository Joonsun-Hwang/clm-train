#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import evaluate
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from dataset import CausalDataset


def test_epoch(args, test_loader, model, tokenizer, metrics):
    model.eval()
    total_loss = 0

    for i, batch in enumerate(
            tqdm(test_loader,
                 desc=' - (Test)      ',
                 leave=False,
                 mininterval=1,
                 file=sys.stdout,
                 disable=not args.accelerator.is_local_main_process)):
        # Fetch inputs and labels
        labels = batch.pop('labels')
        inputs = batch

        # Forward inputs and calculate loss
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=labels)

        # Get predictions and references
        predictions = outputs.logits.argmax(dim=-1)

        # Make predictions and references to sentences from token ids
        # TODO: Generalize to mini batch
        references = labels[(labels != -100)].unsqueeze(0)
        predictions = predictions[:, -references.size(1):]

        references = args.accelerator.pad_across_processes(
            references,
            dim=len(references.size()) - 1,
            pad_index=tokenizer.pad_token_id)
        predictions = args.accelerator.pad_across_processes(
            predictions,
            dim=len(predictions.size()) - 1,
            pad_index=tokenizer.pad_token_id)
        predictions, references = args.accelerator.gather(
            (predictions.contiguous(), references.contiguous()))

        predictions = tokenizer.batch_decode(predictions,
                                             skip_special_tokens=True)
        references = tokenizer.batch_decode(references,
                                            skip_special_tokens=True)

        # Calculate scores
        results = {}
        for key, metric in metrics.items():
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            if key == 'bertscore':
                results[key] = metric.compute(lang='others')
                del results[key]['hashcode']
                results[key] = {
                    k: sum(v) / len(v)
                    for k, v in results[key].items()
                }
            else:
                results[key] = metric.compute()

        batch_loss = args.accelerator.gather(outputs.loss)
        total_loss += torch.sum(batch_loss).item()
    return total_loss / len(test_loader), results


def test(args):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        num_added_toks = tokenizer.add_special_tokens(args.special_tokens_dict)

    # Datasets
    causal_dataset = CausalDataset(args, tokenizer)
    test_loader = causal_dataset.get_dataloaders('test')

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

    # Metrics
    with args.accelerator.main_process_first():
        metrics = {
            'bertscore': evaluate.load('bertscore'),
            'meteor': evaluate.load('meteor'),
        }

    test_loader, model = args.accelerator.prepare(test_loader, model)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)

    # Test
    args.accelerator.wait_for_everyone()
    args.accelerator.print('\n\n[-] Start testing the model\n')

    _, scores = test_epoch(args, test_loader, model, tokenizer, metrics)

    # Print results
    for key, score in scores.items():
        args.accelerator.print(' - ', key, '\n', score)


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

    # Testing Parameters
    parser.add_argument('--batch_size', type=int, default=1)

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
    args.special_tokens_dict = {
        'additional_special_tokens': ['<|User|>', '<|AI|>']
    }

    logging.set_verbosity_error()

    test(args)


if __name__ == '__main__':
    main()
