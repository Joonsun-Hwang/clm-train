import argparse
import json
import os
import shutil
from glob import glob

import nvidia_smi
from accelerate import DistributedType
from transformers import AutoModelForCausalLM, AutoTokenizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mkdir(dir_name: str):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def rmdir(dir_name: str):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)


def rmfile(file_name: str):
    fpaths = glob(file_name)
    for fpath in fpaths:
        if os.path.exists(fpath):
            os.remove(fpath)


def calc_gpu_free_memory(gpu_indices, extra_memory) -> dict:
    nvidia_smi.nvmlInit()

    free_memory = dict()
    for idx in gpu_indices:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(idx)
        gpu_memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        each_free_memory = int((gpu_memory_info.free - extra_memory) /
                               int(os.environ['LOCAL_WORLD_SIZE']))
        if each_free_memory > 0:
            free_memory[idx] = each_free_memory

    return free_memory


def load_checkpoint(args, model, name, epoch):
    ckpt_dir = os.path.join(args.checkpoint_dir, 'checkpoint')
    if not os.path.isdir(os.path.join(ckpt_dir, name)):
        raise ValueError(
            '[!] You should input appropriate checkpoint name at argument "name"'
        )

    with open(os.path.join('checkpoint', name, str(epoch),
                           'config_args.json'),
              'r',
              encoding='utf-8-sig') as i:
        args.__dict__.update(json.load(i))
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join('checkpoint', name, str(epoch)))

    if args.accelerator.distributed_type == DistributedType.DEEPSPEED:
        _ = model.load_checkpoint(os.path.join(ckpt_dir, name), epoch)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(ckpt_dir, name, str(epoch)))
    return args, model, tokenizer


def save_checkpoint(args, model, tokenizer, name):
    ckpt_dir = os.path.join(args.checkpoint_dir, 'checkpoint')
    mkdir(ckpt_dir)

    if args.accelerator.is_main_process:
        tokenizer.save_pretrained(
            os.path.join(ckpt_dir, name, str(args.current_epoch)))
        with open(os.path.join(ckpt_dir, name, str(args.current_epoch),
                               'config_args.json'),
                  'w',
                  encoding='utf-8-sig') as o:
            json.dump(
                {
                    k: args.__dict__[k]
                    for k in args.__dict__ if k != 'accelerator'
                    and k != 'device' and 'config' not in k
                },
                o,
                ensure_ascii=False)

    if args.accelerator.distributed_type == DistributedType.DEEPSPEED:
        args.accelerator.save_state(
            os.path.join(ckpt_dir, name, str(args.current_epoch)))
        success = model.save_checkpoint(os.path.join(ckpt_dir, name),
                                        args.current_epoch)
    else:
        unwrapped_model = args.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(ckpt_dir, name, str(args.current_epoch)),
            is_main_process=args.accelerator.is_main_process,
            save_function=args.accelerator.save,
            state_dict=args.accelerator.get_state_dict(model),
        )


def load_best_checkpoint(args, name):
    ckpt_dir = os.path.join(args.checkpoint_dir, 'checkpoint')
    if not os.path.isdir(os.path.join(ckpt_dir, name)):
        raise ValueError(
            '[!] You should input appropriate checkpoint name at argument "name"'
        )

    with open(os.path.join(ckpt_dir, 'BEST_' + name, 'config_args.json'),
              'r',
              encoding='utf-8-sig') as i:
        args.__dict__.update(json.load(i))
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(ckpt_dir, 'BEST_' + name))
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(ckpt_dir, 'BEST_' + name))
    return args, model, tokenizer


def save_best_checkpoint(args, model, tokenizer, name):
    ckpt_dir = os.path.join(args.checkpoint_dir, 'checkpoint')
    mkdir(ckpt_dir)

    if args.accelerator.is_main_process:
        tokenizer.save_pretrained(os.path.join(ckpt_dir, 'BEST_' + name))
        with open(os.path.join(ckpt_dir, 'BEST_' + name, 'config_args.json'),
                  'w',
                  encoding='utf-8-sig') as o:
            json.dump(
                {
                    k: args.__dict__[k]
                    for k in args.__dict__ if k != 'accelerator'
                    and k != 'device' and 'config' not in k
                },
                o,
                ensure_ascii=False)

    unwrapped_model = args.accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(ckpt_dir, 'BEST_' + name),
        is_main_process=args.accelerator.is_main_process,
        save_function=args.accelerator.save,
        state_dict=args.accelerator.get_state_dict(model),
    )
