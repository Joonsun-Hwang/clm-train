#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time

import accelerate
import evaluate
import numpy as np
import torch
import torch.nn as nn
from accelerate import (Accelerator, DistributedType, infer_auto_device_map,
                        init_empty_weights, load_checkpoint_and_dispatch)
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForPreTraining, AutoTokenizer,
                          get_scheduler, logging)

from dataset import CausalDataset
from utils import calc_gpu_free_memory, save_checkpoint


def train_epoch(args, train_loader, model, optimizer, scheduler):
    model.train()
    total_loss = 0

    for step, batch in enumerate(
            tqdm(train_loader,
                 desc=' - (Training)  ',
                 leave=False,
                 mininterval=1,
                 file=sys.stdout,
                 disable=not args.accelerator.is_local_main_process)):
        # Fetch inputs and labels
        inputs = dict(map(lambda x: (x[0], x[1][:, :-1]), batch.items()))
        labels = batch.input_ids[:, 1:]

        # TODO: Debug for cuda device error.
        if args.model_parallel:
            inputs = dict(
                map(
                    lambda x:
                    (x[0], x[1].to(list(args.device_map.values())[0])),
                    inputs.items()))
            labels = labels.to(list(args.device_map.values())[-1])

        # Forward inputs and calculate loss
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=labels)

        # Calculate gradient
        args.accelerator.backward(outputs.loss)

        # Update model parameters accumulatively
        if step % args.gradient_accumulation_steps == 0:
            args.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                args.grad_clip)
            optimizer.step()
            scheduler.step()

            # Initailize gradient
            optimizer.zero_grad()

        batch_loss = args.accelerator.gather(outputs.loss.to(args.device))
        total_loss += torch.sum(batch_loss).item()
    return total_loss / len(train_loader)


def val_epoch(args, val_loader, model, tokenizer, metrics):
    model.eval()
    total_loss = 0

    for step, batch in enumerate(
            tqdm(val_loader,
                 desc=' - (Validation) ',
                 leave=False,
                 mininterval=1,
                 file=sys.stdout,
                 disable=not args.accelerator.is_local_main_process)):
        # Fetch inputs and labels
        inputs = dict(map(lambda x: (x[0], x[1][:, :-1]), batch.items()))
        labels = batch.input_ids[:, 1:]

        if args.model_parallel:
            inputs = dict(
                map(
                    lambda x:
                    (x[0], x[1].to(list(args.device_map.values())[0])),
                    inputs.items()))
            labels = labels.to(list(args.device_map.values())[-1])

        # Forward inputs and calculate loss
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=labels)

        # Get predictions and references
        predictions = outputs.logits.argmax(dim=-1)
        if args.model_parallel:
            predictions = predictions.to(list(args.device_map.values())[0])
            labels = labels.to(list(args.device_map.values())[0])

        # Make predictions and references to sentences from token id
        predictions = args.accelerator.pad_across_processes(
            predictions,
            dim=len(predictions.size()) - 1,
            pad_index=tokenizer.pad_token_id)
        labels = args.accelerator.pad_across_processes(
            labels,
            dim=len(labels.size()) - 1,
            pad_index=tokenizer.pad_token_id)
        predictions, references = args.accelerator.gather(
            (predictions.contiguous(), labels.contiguous()))

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
            results[key] = metric.compute()

        batch_loss = args.accelerator.gather(outputs.loss)
        total_loss += torch.sum(batch_loss).item()
    return total_loss / len(val_loader), results


def train(args):
    args.current_epoch = 0
    args.current_iter = 0

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        num_added_toks = tokenizer.add_special_tokens(args.special_tokens_dict)

    # Datasets
    with args.accelerator.main_process_first():
        causal_dataset = CausalDataset(args, tokenizer)
    train_loader = causal_dataset.get_dataloaders('train')
    val_loader = causal_dataset.get_dataloaders('validation')

    args.config = AutoConfig.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Model
    if args.model_parallel:
        with init_empty_weights():
            if args.model_type == 'CausalLM':
                model = AutoModelForCausalLM.from_config(args.config)
            elif args.model_type == 'ConditionalGeneration':
                model = AutoModelForPreTraining.from_config(args.config)

        names_module_classes = set()
        for module in model.modules():
            if type(module) == torch.nn.ModuleList:
                names_module_classes.update([
                    submodule.__class__.__name__
                    for submodule in module.children()
                ])

        free_memory = calc_gpu_free_memory(
            list(range(torch.cuda.device_count())), args.extra_memory)
        args.device_map = infer_auto_device_map(
            model,
            max_memory=free_memory,
            no_split_module_classes=list(names_module_classes),
            dtype=torch.float32)

        if args.model_type == 'CausalLM':
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrained_model,
                revision=args.revision,
                device_map=args.device_map,
                cache_dir=os.environ['TRANSFORMERS_CACHE'])
        elif args.model_type == 'ConditionalGeneration':
            model = AutoModelForPreTraining.from_pretrained(
                args.pretrained_model,
                revision=args.revision,
                device_map=args.device_map,
                cache_dir=os.environ['TRANSFORMERS_CACHE'])
    else:
        if args.model_type == 'CausalLM':
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrained_model,
                revision=args.revision,
                cache_dir=os.environ['TRANSFORMERS_CACHE'])
        elif args.model_type == 'ConditionalGeneration':
            model = AutoModelForPreTraining.from_pretrained(
                args.pretrained_model,
                revision=args.revision,
                cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        model.resize_token_embeddings(len(tokenizer))

    if args.add_adapter and args.saved_model:
        args.adapter_name = model.load_adapter(
            os.path.join('checkpoint', 'BEST_adapter_' + args.saved_model))
        model.train_adapter(args.adapter_name)
        args.accelerator.print('[!] Saved checkpoint is loaded')
    elif args.saved_model:
        model.load_state_dict(
            torch.load(
                os.path.join('checkpoint',
                             'BEST_' + args.saved_model + '.ckpt')))
        args.accelerator.print('[!] Saved checkpoint is loaded')
    elif args.add_adapter:
        args.adapter_name = args.checkpoint
        model.add_adapter(args.adapter_name)
        model.train_adapter(args.adapter_name)

    if args.accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Loss and Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer_cls = (
        torch.optim.AdamW
        if args.accelerator.state.deepspeed_plugin is None or "optimizer"
        not in args.accelerator.state.deepspeed_plugin.deepspeed_config else
        accelerate.utils.DummyOptim)
    optimizer = optimizer_cls(optimizer_grouped_parameters,
                              lr=args.learning_rate)

    if args.accelerator.state.distributed_type == DistributedType.DEEPSPEED:
        scheduler = accelerate.utils.DummyScheduler(
            optimizer,
            total_num_steps=(len(train_loader) * args.max_epoch) //
            args.gradient_accumulation_steps,
            warmup_num_steps=args.num_warmup_steps)
    else:
        scheduler = get_scheduler(
            name=args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=(len(train_loader) * args.max_epoch) //
            args.gradient_accumulation_steps)

    # Metrics for validation
    with args.accelerator.main_process_first():
        metrics = {
            'meteor': evaluate.load('meteor'),
        }

    # Prepare everything with accelerator
    if not args.model_parallel:
        model, optimizer, scheduler, train_loader, val_loader = args.accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)
    args.accelerator.register_for_checkpointing(model, optimizer)

    # Train
    args.accelerator.wait_for_everyone()
    args.accelerator.print('\n\n[-] Start training the model\n')
    start_epoch = args.current_epoch
    for epoch in range(start_epoch, args.max_epoch):

        # Train Steps
        args.accelerator.print('[', epoch, '/', args.max_epoch, ']')
        train_start_time = time.time()
        train_loss = train_epoch(args, train_loader, model, optimizer,
                                 scheduler)
        args.accelerator.print(' - Train Time:',
                               round(time.time() - train_start_time, 4))

        # Validation Steps
        val_start_time = time.time()
        val_loss, scores = val_epoch(args, val_loader, model, tokenizer,
                                     metrics)
        args.accelerator.print(' - Validation Time:',
                               round(time.time() - val_start_time, 4))

        # Print results
        args.accelerator.print(' - Train loss:', round(train_loss, 4),
                               '\n - Validation loss:', round(val_loss, 4))
        for key, score in scores.items():
            args.accelerator.print(' - ', key, '\n', score)

        # Append history
        args.train_losses.append(train_loss)
        args.val_losses.append(val_loss)
        args.val_scores.append(scores)

        # Save checkpoint
        args.accelerator.wait_for_everyone()
        if args.accelerator.is_local_main_process:
            save_checkpoint(args, model)

        # Early Stopping
        args.current_epoch += 1
        if args.waiting > args.patient:
            sys.exit(0)

        args.accelerator.print('\n\n')


def main():
    # TODO: There is a bug with using specific devices through CUDA_VISIBLE_DEVICES when distirbuted type is GPU with model parallel and deepspeed
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--cache_root_dir', type=str, default='huggingface')
    parser.add_argument('--max_len', type=int, default=1024)

    # Model Parameters
    parser.add_argument('--mixed_precision',
                        type=str,
                        default='no',
                        choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--model_type', type=str, default='CausalLM')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='EleutherAI/polyglot-ko-1.3b')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--add_adapter', action='store_true')
    parser.add_argument('--saved_model', type=str, default=None)
    parser.add_argument('checkpoint', type=str)

    # Training Parameters
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_batch_size_per_gpu', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='BCE')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--scheduler_type', type=str, default='linear')
    parser.add_argument('--grad_clip', type=float, default=10.)
    parser.add_argument('--patient', type=int, default=3)

    # Process Parameters
    parser.add_argument('--model_parallel', action='store_true')
    parser.add_argument('--extra_memory', type=float, default=4.5e+10)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()

    args.train_losses = []
    args.train_scores = []
    args.val_losses = []
    args.val_scores = []
    args.waiting = 0

    if 'LOCAL_WORLD_SIZE' not in os.environ:  # the number of processes
        os.environ['LOCAL_WORLD_SIZE'] = '1'

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

    # Reproduciblity
    accelerate.utils.set_seed(args.random_seed)

    if args.accelerator.state.distributed_type != DistributedType.MULTI_GPU and args.model_parallel:
        raise ValueError(
            'If you want use "model parallel", your distributed type should be "multi_gpu".'
        )
    if args.model_parallel and torch.cuda.device_count() < 2:
        raise ValueError(
            'If you want use "model parallel", the total number of machines per node should be larger than 1.'
        )
    if args.model_parallel and int(os.environ['LOCAL_WORLD_SIZE']) > 1:
        raise ValueError(
            'If you want use "model parallel", the total number of processes per node should be less than or equal to 1.'
        )

    args.gradient_accumulation_steps = 1
    if args.accelerator.state.deepspeed_plugin is not None:
        args.gradient_accumulation_steps = args.accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"]
    if args.batch_size > args.max_batch_size_per_gpu and args.accelerator.distributed_type != DistributedType.TPU:
        args.gradient_accumulation_steps = args.batch_size // args.max_batch_size_per_gpu
        args.batch_size = args.max_batch_size_per_gpu

    if not args.model_parallel:
        args.extra_memory = 0

    # Additional special tokens
    args.special_tokens_dict = {'additional_special_tokens': ['User:', 'AI:']}

    logging.set_verbosity_error()

    train(args)


if __name__ == '__main__':
    main()
