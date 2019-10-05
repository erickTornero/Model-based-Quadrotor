import json

import torch

from mbrl.mpc import RandomShooter, CrossEntropyMethod


def DecodeMPC(name_mpc:str):
    if name_mpc is 'RandomShooter':
        return RandomShooter
    elif name_mpc is 'CEM':
        return CrossEntropyMethod
    elif name_mpc is 'PDDM':
        return 'PDDM'
    return None
def EncodeMPC(obj):
    if obj is not None:
        return obj.__name__

def DecodeActFunction(name_actfn:str):
    if name_actfn is torch.tanh.__name__:
        return torch.tanh
    elif name_actfn is torch.relu.__name__:
        return torch.relu

def EncodeActFunction(obj):
    if obj is not None:
        return obj.__name__
    