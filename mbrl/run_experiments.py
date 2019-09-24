from mbrl.network import Dynamics
from mbrl.parallel_env import ParallelVrepEnv
from mbrl.runner import Runner
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.mpc import RandomShooter
from mbrl.train_mb import Trainer

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from IPython.core.debugger import set_trace

"""*****************************************
    Hyper-Parameters Settings
********************************************
"""
""" MPC Controller - Random Shooting """
horizon     =   10
candidates  =   1000
discount    =   0.999

""" Environment Setting & runner """
max_path_length         =   250
total_tsteps_per_run    =   10000

""" Training Parameters """
batch_size              =   500
n_epochs                =   80
validation_percent      =   0.2
learning_rate           =   1e-3

""" General parameters """
n_iterations            =   16

"""************************
    Objects for training
***************************
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env_ = QuadrotorEnv(port=27001)
vecenv=ParallelVrepEnv(ports=[25001,28001], max_path_length=250, envClass=QuadrotorEnv)

state_shape= env_.observation_space.shape
action_shape=env_.action_space.shape

dyn = Dynamics(state_shape, action_shape, stack_n=4, sthocastic=False)


dyn = dyn.to(device)

optimizer               =   optim.Adam(lr=learning_rate, params=dyn.parameters())

rs = RandomShooter(horizon, candidates, env_, dyn, device, discount)

trainer =   Trainer(dyn, batch_size, n_epochs, validation_percent, learning_rate, device, optimizer)

print('--------- Creation of runner--------')

runner = Runner(vecenv, env_, dyn, rs, max_path_length, total_tsteps_per_run)

for n_it in range(1, n_iterations+1):
    paths   =   runner.run(random=True) if n_it==1 else runner.run()
    #set_trace()
    observations    =   paths['observations']
    actions         =   paths['actions']
    delta_obs       =   paths['delta_obs']
    data_x          =   np.concatenate((observations, actions), axis=1)
    trainer.fit(data_x, delta_obs)
    print('-------------Info Iteration {}-------------'.format(n_it))
    rolls_info      =   vecenv.get_reset_nrollouts()
    print('Rolls per env> {}, total rollouts {}'.format(rolls_info, sum(rolls_info)))


print('running...')

#paths = runner.run(random=False)
#rolls = vecenv.get_reset_nrollouts()
#print('Rolls> {} \t-->{}'.format(rolls, sum(rolls)))
#
#set_trace()

print(dyn)
