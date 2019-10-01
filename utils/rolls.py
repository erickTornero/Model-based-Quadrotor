from mbrl.mpc import RandomShooter
from mbrl.network import Dynamics
from mbrl.wrapped_env import QuadrotorEnv
from mbrl.runner import StackStAct

import numpy as np
import joblib
import os
import glob

def rollouts(dynamics:Dynamics, env:QuadrotorEnv, mpc:RandomShooter, n_rolls=20, max_path_length=250, save_paths=None):
    """ Generate rollouts for testing & Save paths if it is necessary"""
    nstack  =   dynamics.stack_n
    paths   =   []

    if save_paths is not None:
        pkls    =   glob.glob(os.path.join(save_paths, '*.pkl'))
        assert len(pkls) == 0, "Selected directory is busy, please select other"
        log_path    =   os.path.join(save_paths, 'log.txt')
        texto   =   'Prepare for save paths in "{}"\n'.format(save_paths)

        print('Prepare for save paths in "{}"'.format(save_paths))
        
    #env.set_targetpos(np.random.uniform(-1.0, 1.0, size=(3,)))
    for i_roll in range(1, n_rolls+1):
        #targetposition  =   np.random.uniform(-1.0, 1.0, size=(3))
        targetposition  =   0.8 * np.ones(3, dtype=np.float32)
        next_target_pos =   targetposition

        env.set_targetpos(targetposition)
        obs = env.reset()
        stack_as = StackStAct(env.action_space.shape, env.observation_space.shape, n=nstack, init_st=obs)
        done = False
        timestep    =   0
        cum_reward  =   0.0


        running_paths=dict(observations=[], actions=[], rewards=[], dones=[], next_obs=[], target=[])

        while not done and timestep < max_path_length:
            
            if timestep == 120:
                next_target_pos  = np.zeros(3, dtype=np.float32)
                env.set_targetpos(next_target_pos)

            action = mpc.get_action(stack_as)
               
            next_obs, reward, done, env_info =   env.step(action)

            stack_as.append(acts=action)
            

            if save_paths is not None:
                observation, action = stack_as.get()
                running_paths['observations'].append(observation.flatten())
                running_paths['actions'].append(action.flatten())
                running_paths['rewards'].append(reward)
                running_paths['dones'].append(done)
                running_paths['next_obs'].append(next_obs)
                running_paths['target'].append(targetposition)

                if done or len(running_paths['rewards']) >= max_path_length:
                    paths.append(dict(
                        observation=np.asarray(running_paths['observations']),
                        actions=np.asarray(running_paths['actions']),
                        rewards=np.asarray(running_paths['rewards']),
                        dones=np.asarray(running_paths['dones']),
                        next_obs=np.asarray(running_paths['next_obs']),
                        target=np.asarray(running_paths['target'])
                    ))
            
            targetposition  =   next_target_pos

            stack_as.append(obs=next_obs)
            cum_reward  +=  reward
            timestep += 1

        newtexto   = '{} rollout, reward-> {} in {} timesteps'.format(i_roll, cum_reward, timestep)
        if save_paths is not None:
            joblib.dump(paths, os.path.join(save_paths, 'paths.pkl'))
            with open(log_path, 'w') as fp:
                texto   +=  newtexto + '\n'
                fp.write(texto)


        print(newtexto)

