#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import accelerate
import evaluate
import torch
from accelerate import Accelerator, DistributedType
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  prepare_model_for_int8_training, set_peft_model_state_dict)
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_scheduler, logging, set_seed)

from dataset import CausalDataset, InstructDataset
from utils import load_checkpoint, save_best_checkpoint, save_checkpoint


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

        # Forward inputs and calculate loss
        outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'])

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

        # Forward inputs and calculate loss
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])

        # Get predictions and references
        predictions = outputs.logits.argmax(dim=-1)
        predictions[predictions < 0] = tokenizer.pad_token_id
        predictions = args.accelerator.pad_across_processes(
            predictions,
            dim=len(predictions.size()) - 1,
            pad_index=tokenizer.pad_token_id)

        references = batch['labels'].clone()
        references[references < 0] = tokenizer.pad_token_id
        references = args.accelerator.pad_across_processes(
            references,
            dim=len(references.size()) - 1,
            pad_index=tokenizer.pad_token_id)

        # Gather across through parallel machines
        predictions, references = args.accelerator.gather(
            (predictions.contiguous(), references.contiguous()))

        # Decode
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
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                              revision=args.revision)
    # tokenizer.pad_token = tokenizer.eos_token

    # Additional special tokens
    args.special_tokens_dict = {'additional_special_tokens': []}
    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        num_added_toks = tokenizer.add_special_tokens(args.special_tokens_dict)

    # Datasets
    with args.accelerator.main_process_first():
        #     causal_dataset = CausalDataset(args, tokenizer)
        instruct_dataset = InstructDataset(args, tokenizer)
    # train_loader = causal_dataset.get_dataloaders('train')
    # val_loader = causal_dataset.get_dataloaders('validation')
    train_loader = instruct_dataset.get_dataloaders('train')
    val_loader = instruct_dataset.get_dataloaders('validation')

    args.model_config = AutoConfig.from_pretrained(args.pretrained_model,
                                                   revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                 revision=args.revision)

    if args.special_tokens_dict and args.special_tokens_dict[
            'additional_special_tokens']:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        model = prepare_model_for_int8_training(model)

        args.peft_config = LoraConfig(
            task_type=args.model_type,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[
                "query_key_value"
            ]  # Should check linear layer for query and value in attention block
            # Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/*/modeling_*.py
        )
        model = get_peft_model(model, args.peft_config)

    if args.saved_model:
        # save and load args (losses and scores, etc)
        epoch_histories = [
            path.split('/')[-1]
            for path in glob(os.path.join('checkpoint', args.saved_model, '*'))
        ]
        args.current_epoch = max(
            list(
                map(int,
                    [epoch for epoch in epoch_histories if epoch.isdigit()])))
        args, model, tokenizer = load_checkpoint(args, model, args.saved_model,
                                                 args.current_epoch)
        args.accelerator.print('[*] Saved checkpoint is loaded')

    if args.accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Loss and Scheduler
    optimizer_cls = (
        torch.optim.AdamW
        if args.accelerator.state.deepspeed_plugin is None or "optimizer"
        not in args.accelerator.state.deepspeed_plugin.deepspeed_config else
        accelerate.utils.DummyOptim)
    optimizer = optimizer_cls(params=model.parameters(),
                              weight_decay=args.weight_decay,
                              lr=args.learning_rate)

    if args.accelerator.state.distributed_type == DistributedType.DEEPSPEED:
        scheduler = accelerate.utils.DummyScheduler(
            optimizer=optimizer,
            total_num_steps=(len(train_loader) * args.max_epoch) //
            args.gradient_accumulation_steps,
            warmup_num_steps=args.num_warmup_steps)
    elif args.accelerator.state.distributed_type == DistributedType.MEGATRON_LM:
        scheduler = accelerate.utils.MegatronLMDummyScheduler(
            optimizer=optimizer,
            total_num_steps=(len(train_loader) * args.max_epoch) //
            args.gradient_accumulation_steps,
            warmup_num_steps=args.num_warmup_steps,
        )
    else:
        scheduler = get_scheduler(
            name=args.scheduler_type,
            optimizer=optimizer,
            num_training_steps=(len(train_loader) * args.max_epoch) //
            args.gradient_accumulation_steps,
            num_warmup_steps=args.num_warmup_steps)

    # Metrics for validation
    with args.accelerator.main_process_first():
        metrics = {
            'meteor': evaluate.load('meteor'),
        }

    # Prepare everything with accelerator
    model, optimizer, scheduler, train_loader, val_loader = args.accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)

    # Train
    args.accelerator.wait_for_everyone()
    args.accelerator.print('\n\n[-] Start training the model\n')
    for epoch in range(args.current_epoch, args.max_epoch):

        # Train Steps
        args.accelerator.print('[', epoch, '/', args.max_epoch, ']')
        train_start_time = time.time()
        train_loss = train_epoch(args, train_loader, model, optimizer,
                                 scheduler)
        args.accelerator.print(' - Train Time:',
                               round(time.time() - train_start_time, 4))
        args.accelerator.print(' - Train loss:', round(train_loss, 4))

        # Validation Steps
        val_start_time = time.time()
        val_loss, scores = val_epoch(args, val_loader, model, tokenizer,
                                     metrics)
        args.accelerator.print(' - Validation Time:',
                               round(time.time() - val_start_time, 4))
        args.accelerator.print(' - Validation loss:', round(val_loss, 4))

        # Print results
        args.accelerator.print(' - Scores:')
        for key, score in scores.items():
            args.accelerator.print('\t', score)
        args.accelerator.print()

        # Append history
        args.train_losses.append(train_loss)
        args.val_losses.append(val_loss)
        args.val_scores.append(scores['meteor']['meteor'])

        # Save checkpoint
        save_time = time.time()
        args.accelerator.wait_for_everyone()
        save_checkpoint(args, model, tokenizer, args.checkpoint)
        args.accelerator.print(' - Checkpoint Save Time:',
                               round(time.time() - save_time, 4))

        # Early Stopping
        if len(args.val_losses) > 1 and args.val_losses[-1] < min(args.val_losses[:-1]):
            args.waiting = 0
        else:
            args.waiting += 1
        args.current_epoch += 1

        args.accelerator.wait_for_everyone()
        if args.waiting > args.patient:
            args.accelerator.print('[!] Early stop for training')
            break

        args.accelerator.print('\n\n')

    # Make Best Checkpoint
    args.accelerator.print('[-] Start load and save the best checkpoint')
    save_time = time.time()

    args.best_epoch = args.val_losses.index(min(args.val_losses))
    args.accelerator.print(args.best_epoch, args.val_losses)
    args, model, tokenizer = load_checkpoint(args, model, args.checkpoint,
                                             args.best_epoch)
    args.accelerator.print(args.best_epoch, args.val_losses)
    save_best_checkpoint(args, model, tokenizer, args.checkpoint)

    args.accelerator.print(' - Best Checkpoint Save Time:',
                           round(time.time() - save_time, 4))
    args.accelerator.print(
        '[*] The best checkpoint was saved and finish training')
    args.accelerator.print('\n\n')


def main():
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--max_len', type=int, default=2048)

    # Model Parameters
    parser.add_argument(
        '--mixed_precision',
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.")
    parser.add_argument('--model_type', type=str, default='CAUSAL_LM')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='beomi/KoAlpaca-Polyglot-5.8B')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--saved_model', type=str, default=None)
    parser.add_argument('checkpoint', type=str)

    # Training Parameters
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_batch_size_per_gpu', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--scheduler_type', type=str, default='linear')
    parser.add_argument('--grad_clip', type=float, default=10.)
    parser.add_argument('--patient', type=int, default=3)

    # PEFT parameters
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=int, default=0.05)
    # parser.add_argument('--use_ptuning', action='store_true')

    # Process Parameters
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()

    args.train_losses = []
    args.train_scores = []
    args.val_losses = []
    args.val_scores = []
    args.waiting = 0

    os.environ[
        "TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism deadlock warning

    if 'LOCAL_WORLD_SIZE' not in os.environ:  # the number of processes
        os.environ['LOCAL_WORLD_SIZE'] = '1'

    # Accelerator
    args.accelerator = Accelerator(cpu=args.cpu,
                                   mixed_precision=args.mixed_precision)
    args.device = args.accelerator.device

    # Reproduciblity
    set_seed(args.random_seed)

    args.gradient_accumulation_steps = 1
    if args.accelerator.state.deepspeed_plugin is not None:
        args.gradient_accumulation_steps = args.accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"]
    if args.batch_size > args.max_batch_size_per_gpu and args.accelerator.distributed_type != DistributedType.TPU:
        args.gradient_accumulation_steps = args.batch_size // (
            args.max_batch_size_per_gpu * int(os.environ['LOCAL_WORLD_SIZE']))
        args.batch_size = args.max_batch_size_per_gpu

    logging.set_verbosity_error()

    train(args)


if __name__ == '__main__':
    main()
