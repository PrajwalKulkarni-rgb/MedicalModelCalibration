# import logging
# from .logger import *
# from .eval import *
# from argparsor import parse_args
# from .misc import *
# from .eval import EarlyStopping
import torch
import os
import shutil
from .metrics import CalibrationMetrics


def get_lr(optimizer):
    """Returns the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    """
    Saves the current model state.
    If the model is the best so far, also saves a 'model_best.pth' copy.
    """
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth"))

def create_save_path(args):
    """
    Generates a string representing the model and loss function configuration.
    Used for naming save directories or log files.
    """
    save_str = f"_{args.model}_{args.loss}"

    if args.loss in ["MDCA", "DCA", "MMCE"]:
        save_str += f"_beta={args.beta}"
    elif args.loss in ["FLSD", "focal_loss", "FL"]:
        save_str += f"_gamma={args.gamma}"
    elif "LS" in args.loss:
        save_str += f"_alpha={args.alpha}"

    return save_str
# -*- coding: utf-8 -*-

