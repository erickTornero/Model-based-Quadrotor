from mbrl.network import Dynamics
from mbrl.parallel_env import ParallelVrepEnv
from mbrl.runner import Runner
from wrapper_quad.wrapper_vrep import VREPQuad

from IPython.core.debugger import set_trace

import torch

env_ = VREPQuad(port=27001)
vecenv=ParallelVrepEnv(ports=[25001,26001],num_rollouts=10, max_path_length=250, envClass=VREPQuad)

state_shape= env_.observation_space.shape
action_shape=env_.action_space.shape

dyn = Dynamics(state_shape, action_shape, stack_n=4, sthocastic=True)

x = torch.randn(88)
x = dyn(x)

print('--------- Creation of runner--------')

runner = Runner(vecenv, env_, dyn, None, 250, 100)

print('running...')

paths = runner.run(random=True)

set_trace()
print(x)

print(dyn)
