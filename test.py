#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import evaluate
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          logging)

from accelerate import Accelerator
from dataset import CausalDataset
from model import GPTNeoXPrefixForCausalLM
from utils import str2bool


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
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                              revision=args.revision)

    # Additional special tokens
    args.special_tokens_dict = {'additional_special_tokens': []}
    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        num_added_toks = tokenizer.add_special_tokens(args.special_tokens_dict)

    # Datasets
    causal_dataset = CausalDataset(args, tokenizer)
    test_loader = causal_dataset.get_dataloaders('test')

    # Model
    if args.p_tuning:
        args.config.pre_seq_len = args.pre_seq_len
        args.config.prefix_projection = args.prefix_projection
        args.config.prefix_hidden_size = args.prefix_hidden_size
        args.config.hidden_dropout_prob = args.hidden_dropout_prob
        model = GPTNeoXPrefixForCausalLM.from_pretrained(
            args.pretrained_model, revision=args.revision, config=args.config)
    elif args.model_type == 'CausalLM':
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                     revision=args.revision)

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

    args = parser.parse_args()

    # Accelerator
    args.accelerator = Accelerator(cpu=args.cpu,
                                   mixed_precision=args.mixed_precision)
    args.device = args.accelerator.device
    logging.set_verbosity_error()

    test(args)


if __name__ == '__main__':
    main()
