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
import os
import joblib
from utils.plots import *
from IPython.core.debugger import set_trace

"""*****************************************
    Hyper-Parameters Settings
********************************************
"""
""" MPC Controller - Random Shooting """
horizon     =   10
candidates  =   1000
discount    =   0.99

""" Environment Setting & runner """
max_path_length         =   250
total_tsteps_per_run    =   10000

""" Training Parameters """
batch_size              =   500
n_epochs                =   100
validation_percent      =   0.2
learning_rate           =   1e-4

""" General parameters """
id_executor             =   'sample2'
n_iterations            =   24
save_path               =   os.path.join('./data/', id_executor)

"""************************
    Objects for training
***************************
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env_ = QuadrotorEnv(port=27001) # 28
#vecenv=ParallelVrepEnv(ports=[25001,28001], max_path_length=250, envClass=QuadrotorEnv)
vecenv=ParallelVrepEnv(ports=[19999, 20001,21001,22001], max_path_length=250, envClass=QuadrotorEnv)
state_shape= env_.observation_space.shape
action_shape=env_.action_space.shape

dyn = Dynamics(state_shape, action_shape, stack_n=4, sthocastic=False)


dyn = dyn.to(device)

optimizer               =   optim.Adam(lr=learning_rate, params=dyn.parameters())

rs = RandomShooter(horizon, candidates, env_, dyn, device, discount)

trainer =   Trainer(dyn, batch_size, n_epochs, validation_percent, learning_rate, device, optimizer)

print('--------- Creation of runner--------')

runner = Runner(vecenv, env_, dyn, rs, max_path_length, total_tsteps_per_run)

if not os.path.exists(save_path):
    os.makedirs(save_path)

mean_reward_maximum =   0.0
for n_it in range(1, n_iterations+1):
    print('============================================')
    print('\t\t Iteration {} \t\t\t'.format(n_it))
    print('============================================')
    paths   =   runner.run(random=True) if n_it==1 else runner.run()
    #set_trace()
    observations    =   paths['observations']
    actions         =   paths['actions']
    delta_obs       =   paths['delta_obs']
    total_rewards   =   paths['rewards']
    data_x          =   np.concatenate((observations, actions), axis=1)
    tr_loss, vl_loss = trainer.fit(data_x, delta_obs)
    print('-------------Info {}-------------'.format(n_it))
    rolls_info      =   vecenv.get_reset_nrollouts()
    print('Rolls per env> {}, total rollouts {}'.format(rolls_info, sum(rolls_info)))
    print('total time steps: \t{}'.format(actions.shape[0]))
    print('Reward mean: \t\t{}'.format(np.mean(total_rewards)))
    print('Reward  std: \t\t{}'.format(np.std(total_rewards)))
    print('Reward  min: \t\t{}'.format(np.min(total_rewards)))
    print('Reward  max: \t\t{}'.format(np.max(total_rewards)))
    
    print('Saving model ...')
    if mean_reward_maximum < np.mean(total_rewards): 
        mean_reward_maximum = np.mean(total_rewards)
        torch.save(dyn.state_dict(), os.path.join(save_path, 'params_high.pkl'))
    else:
        torch.save(dyn.state_dict(), os.path.join(save_path, 'params.pkl'))

    joblib.dump(observations, os.path.join(save_path, 'observations_it_' + str(n_it)+'.pkl'))
    plot_loss_per_iteration(tr_loss, vl_loss, os.path.join(save_path, 'loss_it_'+str(n_it)+'.png'))

#paths = runner.run(random=False)
#rolls = vecenv.get_reset_nrollouts()
#print('Rolls> {} \t-->{}'.format(rolls, sum(rolls)))
#
#set_trace()

print(dyn)
