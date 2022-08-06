import argparse
import os
import shutil
import time
from glob import glob

import numpy as np
import nvidia_smi
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


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


def calc_perform(logits, labels, criterion=None):
    # logits: (batch_size, n_class), labels: (batch_size, n_class)
    probs = torch.exp(F.log_softmax(logits, dim=1))
    argmax_labels = torch.argmax(labels, dim=1).tolist()
    argmax_probs = torch.argmax(probs, dim=1).tolist()

    if criterion:
        loss = criterion(logits, labels)

        return loss, argmax_labels, argmax_probs
    else:
        return argmax_labels, argmax_probs


def calc_conf_mat(conf_mat):
    conf_mat = conf_mat + 1e-10  # for smoothing
    precision = np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis=0))
    recall = np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis=1))
    return (2 * precision * recall) / (precision + recall)


def print_perform(header, loss, score):
    print(' - {header:12} loss: {loss:4.4f}, score: {score:3.4f} %\
        '.format(header=f"({header})", loss=min(loss, 999), score=score * 100))


def print_test_result(argmax_labels, argmax_probs, return_score=False):
    conf_mat = confusion_matrix(argmax_labels, argmax_probs)
    print("Confusion Matrix:")
    print(conf_mat)

    accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    precisions = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    recalls = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    print("Accuracy : ", round(accuracy * 100, 4))
    print("Precisions :")
    for i, precision in enumerate(precisions):
        print(i, ":", round(precision * 100, 4))
    print("Recalls :")
    for i, recall in enumerate(recalls):
        print(i, ":", round(recall * 100, 4))

    f1_score = 2*np.mean(precisions)*np.mean(recalls)/\
        (np.mean(precisions)+np.mean(recalls))
    print("F1 Score :", round(f1_score, 4))

    if return_score:
        return precisions, recalls, f1_score


def save_checkpoint(args: argparse.Namespace, model_state, classifier_state, optimizer_state):
    checkpoint = {
        'args': args,
        'model_state': model_state,
        'classifier_state': classifier_state,
        'optimizer_state': optimizer_state
    }

    ckpt_dir = os.path.join(args.data_root_dir, 'checkpoint')
    mkdir(ckpt_dir)

    file_name = args.checkpoint + '.ckpt'
    torch.save(checkpoint, os.path.join(ckpt_dir, file_name))

    args.waiting += 1

    if args.val_losses[-1] <= min(args.val_losses):
        args.waiting = 0
        file_name = 'BEST_' + file_name
        torch.save(checkpoint, os.path.join(ckpt_dir, file_name))
        print('\t[!] The best checkpoint is updated.')
