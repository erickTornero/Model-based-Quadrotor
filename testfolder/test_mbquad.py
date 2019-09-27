from mbrl.network import Dynamics
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.mpc import RandomShooter

from utils.rolls import rollouts
import os
import torch

from IPython.core.debugger import set_trace

restore_folder='./data/sample5/'
env_    =   QuadrotorEnv(port=28001)
state_shape =   env_.observation_space.shape
action_shape=   env_.action_space.shape

horizon     =   20
candidates  =   1500
discount    =   0.99
nstack      =   4
max_path_length =   250
nrollouts   =   10

device      =   torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


dynamics    =   Dynamics(state_shape, action_shape, stack_n=nstack, sthocastic=False)
rs          =   RandomShooter(horizon, candidates, env_, dynamics, device, discount)
checkpoint  =   torch.load(os.path.join(restore_folder, 'params_high.pkl'))
dynamics.load_state_dict(checkpoint['model_state_dict'])

dynamics.mean_input =   checkpoint['mean_input']
dynamics.std_input  =   checkpoint['std_input']
dynamics.epsilon    =   checkpoint['epsilon']

dynamics.to(device)



rollouts(dynamics, env_, rs, nrollouts, max_path_length)