import json

import torch

from mbrl.mpc import RandomShooter, CrossEntropyMethod


def DecodeMPC(name_mpc:str):
    if name_mpc == 'RandomShooter':
        return RandomShooter
    elif name_mpc == 'CEM':
        return CrossEntropyMethod
    elif name_mpc == 'PDDM':
        return 'PDDM'
    return None
def EncodeMPC(obj):
    if obj is not None:
        return obj.__name__

def DecodeActFunction(name_actfn:str):
    if name_actfn == torch.tanh.__name__:
        return torch.tanh
    elif name_actfn == torch.relu.__name__:
        return torch.relu
    else: assert True, 'Not found activation function'

def EncodeActFunction(obj):
    if obj is not None:
        return obj.__name__
    