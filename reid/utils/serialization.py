from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil

import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise "=> No checkpoint found at '{}'".format(fpath)

def copy_state_dict_dsbn(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        index = ['bns.0', 'bns.1', 'bns.2', 'bns.3']
        if isinstance(param, Parameter):
            param = param.data
        if 'bns.0' in name:
            for ind in index:
                new_name = name.replace('bns.0', ind)
                tgt_state[new_name].copy_(param)
                copied_names.add(new_name)
        else:
            if name not in tgt_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            if param.size() != tgt_state[name].size():
                print('mismatch:', name, param.size(), tgt_state[name].size())
                continue
            tgt_state[name].copy_(param)
            copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

def load_ckpt(ckpt, model):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            print(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        else:
            v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            print(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model
