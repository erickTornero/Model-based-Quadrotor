from mbrl.network import Dynamics
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.mpc import RandomShooter

from utils.rolls import rollouts
import os
import torch

from IPython.core.debugger import set_trace

restore_folder='./data/sample4/'
env_    =   QuadrotorEnv(port=28001)
state_shape =   env_.observation_space.shape
action_shape=   env_.action_space.shape

horizon     =   15
candidates  =   1000
discount    =   0.99
nstack      =   4

device      =   torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

set_trace()

dynamics    =   Dynamics(state_shape, action_shape, stack_n=nstack, sthocastic=False)
rs          =   RandomShooter(horizon, candidates, env_, dynamics, device, discount)
dynamics.load_state_dict(torch.load(os.path.join(restore_folder, 'params_high.pkl')))


rollouts(dynamics, env_, rs, 10)