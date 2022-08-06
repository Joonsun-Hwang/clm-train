import argparse
import os
import sys
import random
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AdamW, logging
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from datasets import HanyuaDialogDataset
from utils import calc_gpu_free_memory, calc_perform, print_perform, save_checkpoint


def train_epoch(args, train_loader, tokenizer, model, criterion, optimizer):
    model.train()
    sum_train_loss = 0

    print('aaa')

    for i, batch in enumerate(
            tqdm(train_loader,
                 desc=' - (Training)  ',
                 leave=False,
                 mininterval=1,
                 file=sys.stdout,
                 disable=not args.accelerator.is_local_main_process)
    ):      
            # initailize gradient
            optimizer.zero_grad()

            # fetch inputs
            inputs = tokenizer(batch, padding=True, return_tensors='pt').to(0)  # (batch_size, seq_len)
            targets = inputs.input_ids[:, 1:]

            # forward inputs and calculate loss
            outputs = model(**inputs)
            train_loss, _ = calc_perform(outputs.logits, targets)

            # calculate gradient and update model parameters
            args.accelerator.backward(train_loss)
            optimizer.step()

    return sum_train_loss


def val_epoch(args, val_loader, tokenizer, model, criterion):
    model.eval()
    sum_val_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(
                tqdm(val_loader,
                     desc=' - (Validation) ',
                     leave=False,
                     mininterval=1,
                     file=sys.stdout,
                     disable=not args.accelerator.is_local_main_process)
        ):  # keys: ['label', 'input_ids', 'attention_mask', 'token_type_ids', 'cls_token_indices']
            exit()

    return sum_val_loss


def train(args):
    args.current_epoch = 0
    args.current_iter = 0

    args.accelerator = Accelerator()
    config = AutoConfig.from_pretrained(args.pretrained_model, revision=args.revision,
                                    cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])

    # Datasets
    train_dataset = HanyuaDialogDataset(args, tokenizer, 'train')
    val_dataset = HanyuaDialogDataset(args, tokenizer, 'val')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            pin_memory=True)

    # Model
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

    # Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader, model, optimizer = args.accelerator.prepare(
        train_loader, model, optimizer)

    # Train
    if args.accelerator.is_local_main_process:
        print(args)
    
    start_epoch = args.current_epoch
    for epoch in range(start_epoch, args.max_epoch):
        avg_train_loss = train_epoch(args, train_loader, tokenizer, model, criterion,
                                     optimizer)
        avg_val_loss = val_epoch(args, val_loader, tokenizer, model, criterion)
        print(avg_train_loss, avg_val_loss)

        save_checkpoint(args, model.state_dict(), optimizer.state_dict())

        args.current_epoch += 1
        if args.waiting > args.patient:
            break


def main():
    # !torchrun --nproc_per_node 4 train.py
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--data_root_dir', type=str, default='datasets')
    parser.add_argument('--data_dir',
                        type=str,
                        default='hanyua_augmented_dialog')
    parser.add_argument('--cache_root_dir',
                        type=str,
                        default='/home/jsunhwang/huggingface_models')
    parser.add_argument('--max_len', type=int, default=1200)

    # Model Parameters
    parser.add_argument('--pretrained_model',
                        type=str,
                        default='kakaobrain/kogpt')
    parser.add_argument('--revision', type=str, default='KoGPT6B-ryan1.5b')
    parser.add_argument('--saved_model', type=str, default=None)
    parser.add_argument('checkpoint', type=str)

    # Training Parameters
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='BCE')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--patient', type=int, default=3)

    # Multi-process Parameters
    parser.add_argument('--cuda_visible_devices', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--extra_memory', type=float, default=4.5e+10)

    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()

    # args.device = torch.device(
    #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.train_losses = []
    args.train_scores = []
    args.val_losses = []
    args.val_scores = []
    args.waiting = 0

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)  # if use multi-gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # need to set early in shell command 'torchrun --nproc_per_node LOCAL_WORLD_SIZE'
    if 'LOCAL_WORLD_SIZE' not in os.environ:
        os.environ['LOCAL_WORLD_SIZE'] = '1'
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    args.gpu_indices = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(args.cache_root_dir,
                                                    args.pretrained_model,
                                                    args.revision)

    logging.set_verbosity_error()

    train(args)


if __name__ == '__main__':
    main()
