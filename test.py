import argparse
import os
import random
import sys
import time

import accelerate
import evaluate
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from dataset import HanyuaDataset


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
        inputs = dict(map(lambda x: (x[0], x[1][:, :-1]), batch.items()))
        labels = batch.input_ids[:, 1:].to(args.device)

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
    return total_loss / len(test_loader), results


def test(args):
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
    test_loader = hanyua_dataset.get_dataloaders('test')

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        # torch_dtype='auto',  # There is a bug for 'facebook/opt'
        low_cpu_mem_usage=True,
        cache_dir=os.environ['TRANSFORMERS_CACHE'])
    model.load_state_dict(
        torch.load(os.path.join(args.data_root_dir, 'checkpoint',
                                'BEST_' + args.checkpoint + '.ckpt'),
                   map_location=args.device))

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
    parser.add_argument('checkpoint', type=str)

    # Testing Parameters
    parser.add_argument('--batch_size', type=int,
                        default=1)  # 1 process: Max 4; 2 process: Max 1

    # Multi-process Parameters
    parser.add_argument('--mixed_precision',
                        type=str,
                        default='no',
                        choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    os.environ['TRANSFORMERS_CACHE'] = os.path.join(args.cache_root_dir,
                                                    args.pretrained_model,
                                                    args.revision)

    logging.set_verbosity_error()

    test(args)


if __name__ == '__main__':
    main()
