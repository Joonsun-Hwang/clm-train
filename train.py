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
from accelerate import (Accelerator, infer_auto_device_map, init_empty_weights,
                        load_checkpoint_and_dispatch)
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer, logging)

from dataset import HanyuaDataset
from utils import calc_gpu_free_memory, save_checkpoint


def train_epoch(args, train_loader, model, optimizer):
    model.train()
    total_loss = 0

    for i, batch in enumerate(
            tqdm(train_loader,
                 desc=' - (Training)  ',
                 leave=False,
                 mininterval=1,
                 file=sys.stdout,
                 disable=not args.accelerator.is_local_main_process)):
        # Fetch inputs and labels
        inputs = dict(map(lambda x: (x[0], x[1][:, :-1]), batch.items()))
        labels = batch.input_ids[:, 1:].to(args.device_map['lm_head'])

        # forward inputs and calculate loss
        outputs = model(**inputs, labels=labels)

        # calculate gradient and update model parameters
        args.accelerator.backward(outputs.loss)
        args.accelerator.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.grad_clip)
        optimizer.step()

        # initailize gradient
        optimizer.zero_grad()

        batch_loss = args.accelerator.gather(outputs.loss.to(args.device))
        total_loss += torch.sum(batch_loss).item()
    return total_loss / len(train_loader)


def val_epoch(args, val_loader, model, tokenizer, metrics):
    model.eval()
    total_loss = 0

    for i, batch in enumerate(
            tqdm(val_loader,
                 desc=' - (Validation) ',
                 leave=False,
                 mininterval=1,
                 file=sys.stdout,
                 disable=not args.accelerator.is_local_main_process)):
        # Fetch inputs and labels
        inputs = dict(map(lambda x: (x[0], x[1][:, :-1]), batch.items()))
        labels = batch.input_ids[:, 1:].to(args.device_map['lm_head'])

        # forward inputs and calculate loss
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)

        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = args.accelerator.gather(
            (predictions.to(args.device),
             labels.to(args.device)))  # Should load the tensors to gpu:0

        predictions = tokenizer.batch_decode(predictions)
        references = tokenizer.batch_decode(references)

        results = {}
        for key, metric in metrics.items():
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            if key == 'bertscore':
                results[key] = metric.compute(lang='others')
            else:
                results[key] = metric.compute()

        batch_loss = args.accelerator.gather(outputs.loss.to(args.device))
        total_loss += torch.sum(batch_loss).item()
    return total_loss / len(val_loader), results


def train(args):
    args.current_epoch = 0
    args.current_iter = 0

    # Accelerator
    args.accelerator = Accelerator(cpu=args.cpu,
                                   mixed_precision=args.mixed_precision)
    args.device = args.accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Datasets
    hanyua_dataset = HanyuaDataset(args, tokenizer)
    train_loader = hanyua_dataset.get_dataloaders('train')
    val_loader = hanyua_dataset.get_dataloaders('validation')

    # Model
    config = AutoConfig.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    names_module_classes = set()
    for module in model.modules():
        if type(module) == torch.nn.ModuleList:
            names_module_classes.update([
                submodule.__class__.__name__
                for submodule in module.children()
            ])

    free_memory = calc_gpu_free_memory(args.gpu_indices, args.extra_memory)
    args.device_map = infer_auto_device_map(
        model,
        max_memory=free_memory,
        no_split_module_classes=list(names_module_classes),
        dtype=torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        device_map=args.device_map,
        # torch_dtype='auto',  # There is a bug for 'facebook/opt'
        low_cpu_mem_usage=True,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Loss and Metric
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    metrics = {
        'bertscore':
        evaluate.load('bertscore'),
        'meteor':
        evaluate.load('meteor'),
        'bleurt':
        evaluate.load('bleurt',
                      module_type='metric',
                      checkpoint='bleurt-large-512'),
    }

    # train_loader, val_loader, model, optimizer = args.accelerator.prepare(
    #     train_loader, val_loader, model, optimizer)

    # Use accelerator.print to print only on the main process.
    args.accelerator.print('\n\n[-] Arguments:\n')
    args.accelerator.print(args)
    args.accelerator.register_for_checkpointing(model, optimizer)

    # Train
    args.accelerator.wait_for_everyone()
    args.accelerator.print('\n\n[-] Start training the model\n')
    start_epoch = args.current_epoch
    for epoch in range(start_epoch, args.max_epoch):
        args.accelerator.print('[', epoch, '/', args.max_epoch, ']')
        train_loss = train_epoch(args, train_loader, model, optimizer)
        val_loss, scores = val_epoch(args, val_loader, model, tokenizer,
                                     metrics)

        # Print results
        args.accelerator.print('- Train loss:', round(train_loss, 4),
                               '\n - Validation loss:', round(val_loss, 4))
        for key, score in scores.items():
            args.accelerator.print('- ', key, '\n', score)
        args.accelerator.print('\n')

        # Append history
        args.train_losses.append(train_loss)
        args.val_losses.append(val_loss)
        args.val_scores.append(scores)

        # Save checkpoint
        args.accelerator.wait_for_everyone()
        if args.accelerator.is_local_main_process:
            save_checkpoint(args, model)

        args.current_epoch += 1
        if args.waiting > args.patient:
            exit()


def main():
    # !torchrun --nproc_per_node 4 train.py
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_root_dir', type=str, default='data')
    parser.add_argument('--data_dir',
                        type=str,
                        default='hanyua_augmented_dialog')
    parser.add_argument('--cache_root_dir',
                        type=str,
                        default='/home/jsunhwang/huggingface_models')
    parser.add_argument('--max_len', type=int, default=2048)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='kakaobrain/kogpt')
    parser.add_argument('--revision', type=str, default='KoGPT6B-ryan1.5b')
    parser.add_argument('--saved_model', type=str, default=None)
    parser.add_argument('checkpoint', type=str)

    # Training Parameters
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int,
                        default=1)  # 1 process: Max 4; 2 process: Max 1
    parser.add_argument('--loss_type', type=str, default='BCE')
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--patient', type=int, default=3)

    # Multi-process Parameters
    parser.add_argument('--mixed_precision',
                        type=str,
                        default='no',
                        choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--cuda_visible_devices',
                        type=str,
                        default='0,1,2,3,4,5,6,7')
    parser.add_argument('--extra_memory', type=float, default=4.0e+10)

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()

    # args.device = torch.device(
    #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.train_losses = []
    args.train_scores = []
    args.val_losses = []
    args.val_scores = []
    args.waiting = 0

    # Reproduciblity
    # random.seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed)  # if use multi-gpu
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    accelerate.utils.set_seed(args.random_seed)

    # need to set early in shell command 'torchrun --nproc_per_node LOCAL_WORLD_SIZE'
    if 'LOCAL_WORLD_SIZE' not in os.environ:
        os.environ['LOCAL_WORLD_SIZE'] = '1'
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    args.gpu_indices = list(
        map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(args.cache_root_dir,
                                                    args.pretrained_model,
                                                    args.revision)

    logging.set_verbosity_error()

    train(args)


if __name__ == '__main__':
    main()
