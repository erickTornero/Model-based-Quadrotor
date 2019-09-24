from mbrl.network import Dynamics
from mbrl.parallel_env import ParallelVrepEnv
from mbrl.runner import Runner
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.mpc import RandomShooter

import torch

from IPython.core.debugger import set_trace

"""
    Hyper-Parameters Settings
"""
""" MPC Controller - Random Shooting """
horizon     =   10
candidates  =   1000
discount    =   0.999

""" Environment Setting & runner """
max_path_length         =   250
total_tsteps_per_run    =   200


env_ = QuadrotorEnv(port=27001)
vecenv=ParallelVrepEnv(ports=[25001,28001], max_path_length=250, envClass=QuadrotorEnv)

state_shape= env_.observation_space.shape
action_shape=env_.action_space.shape

dyn = Dynamics(state_shape, action_shape, stack_n=4, sthocastic=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dyn = dyn.to(device)

rs = RandomShooter(horizon, candidates, env_, dyn, device, discount)

print('--------- Creation of runner--------')

runner = Runner(vecenv, env_, dyn, rs, max_path_length, total_tsteps_per_run)

print('running...')

paths = runner.run(random=False)
rolls = vecenv.get_reset_nrollouts()
print('Rolls> {} \t-->{}'.format(rolls, sum(rolls)))

set_trace()

print(dyn)
