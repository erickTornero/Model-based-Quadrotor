from mbrl.network import Dynamics
from mbrl.parallel_env import ParallelVrepEnv
from mbrl.runner import Runner
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.mpc import RandomShooter

from IPython.core.debugger import set_trace

import torch

env_ = QuadrotorEnv(port=27001)
vecenv=ParallelVrepEnv(ports=[25001,28001],num_rollouts=10, max_path_length=250, envClass=QuadrotorEnv)

state_shape= env_.observation_space.shape
action_shape=env_.action_space.shape

dyn = Dynamics(state_shape, action_shape, stack_n=4, sthocastic=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


rs = RandomShooter(10, 1000, env_, dyn, device, 0.999)

print('--------- Creation of runner--------')

runner = Runner(vecenv, env_, dyn, rs, 250, 100)

print('running...')

paths = runner.run(random=True)

set_trace()
print(x)

print(dyn)
