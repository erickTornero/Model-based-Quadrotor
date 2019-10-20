
import numpy as np
from collections import deque
from mbrl.data_processor import DataProcessor
from mbrl.mpc import RandomShooter
import itertools
import torch
from tqdm import tqdm
from IPython.core.debugger import set_trace



class Runner:
    """
        Collect Samples of quadrotor
    """

    def __init__(self, vecenv, env, net, mpc:RandomShooter, max_path_len, total_nsteps):
        self.vec_env    =   vecenv
        self.env_   =   env
        self.net    =   net
        self.nstack =   net.stack_n

        self.max_path_len   =   max_path_len
        #self.total_samples  =   n_rollouts * nsteps
        self.total_samples  =   total_nsteps

        self.n_parallel =   self.vec_env.n_parallel

        self.dProcesor       =   DataProcessor(0.99)

        #self.env_   =   self.vec_env.getenv
        self.mpc    =   mpc


    def run(self, random=False):
        
        print('Collecting samples '+ ('Randomly' if random else 'with policy'))
        paths       =   []
        n_samples   =   0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.n_parallel)]

        # Reset environments
        #obses   =   np.asarray(self.vec_env.reset())
        obses   =   self.vec_env.reset()
        stack_as    =   [StackStAct(self.env_.action_space.shape, self.env_.observation_space.shape, n=self.nstack, init_st=ob) for ob in obses]
        
        # TQDM bar
        pbar    =   tqdm(total=self.total_samples)
        while n_samples < self.total_samples:
            if random:
                actions =   np.stack([self.env_.action_space.sample() for _ in range(self.n_parallel)], axis=0)
            else:
                # Get next action given stat of actions and states
                #obs_stack, act_stack = stack_as.get()
                #actions = mpc.get_action(obs_stack, act_stack[1:])
                #actions =   np.stack([self.mpc.get_action_PDDM(stack_, 5.0, 0.6) for stack_ in stack_as], axis=0)
                #actions =   np.stack([self.mpc.get_action(stack_) for stack_ in stack_as], axis=0)
                #actions =   np.stack([self.mpc.get_action_CEM(stack_, 50, 3, 0.8) for stack_ in stack_as], axis=0)
                actions =   np.stack([self.mpc.get_action_torch(stack_) for stack_ in stack_as], axis=0)

            next_obs, rewards, dones, env_infos = self.vec_env.step(actions)

            #from IPython.core.debugger import set_trace
            #set_trace()
            delta_obs   =   [stack_.get_last_state() for stack_ in stack_as]
            delta_obs   =   [next_ob - delta_ob for delta_ob, next_ob in zip(delta_obs, next_obs)]

            _   = [stack_.append(acts=act) for act, stack_ in zip(actions, stack_as)]
            # append new samples:

            new_samples = 0
            for idx, stack_, reward, done, next_ob, delta_ob in zip(itertools.count(), stack_as, rewards, dones, next_obs, delta_obs):
                observation, action =   stack_.get()
                running_paths[idx]['observations'].append(observation.flatten())
                running_paths[idx]['actions'].append(action.flatten())
                running_paths[idx]['rewards'].append(reward)
                running_paths[idx]['dones'].append(done)
                running_paths[idx]['next_obs'].append(next_ob)
                running_paths[idx]['delta_obs'].append(delta_ob)


                if len(running_paths[idx]['rewards']) >= self.max_path_len or done:
                    paths.append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        dones=np.asarray(running_paths[idx]["dones"]),
                        next_obs=np.asarray(running_paths[idx]['next_obs']),
                        delta_obs=np.asarray(running_paths[idx]['delta_obs'])
                    ))
                    new_samples += len(running_paths[idx]['rewards'])
                    running_paths[idx] = _get_empty_running_paths_dict()
                    # Restart environments
                    #obses   =   self.vec_env.reset()
                    ob_    =   self.vec_env.reset_remote(idx)
                    #stack_as    =   [StackStAct(self.env_.action_space.shape, self.env_.observation_space.shape, n=4, init_st=ob) for ob in obses]
                    stack_as[idx].reset_stacks(init_st=ob_)

            n_samples += new_samples
            pbar.update(new_samples)
            ## Update all the next states
            for done, stack_, next_ob in zip(dones, stack_as, next_obs):
                if not done:
                    stack_.append(obs=next_ob)
            
            #[stack_.append(obs=next_ob) for next_ob, stack_ in zip(next_obs, stack_as)]
        pbar.close()
        sampled_data = self.dProcesor.process(paths)

        return sampled_data



        
class StackStAct:
    """
        Stack State-Action class:
        Help to record stack of the current and past
        states & actions
    """
    def __init__(self, act_shape, st_shape, n:int, init_st = None, init_ac = None):
        self.action_shape = act_shape
        self.state_shape = st_shape
        self.n = n

        if init_ac is None: init_ac = np.zeros(act_shape)
        if init_st is None: init_st = np.zeros(st_shape) 
        self.actions_stack =   deque(n * [init_ac], maxlen=n)
        self.states_stack  =   deque(n * [init_st], maxlen=n)

    def append_and_get(self, obs=None, acts=None):
        if obs is not None: self.states_stack.append(obs)
        if acts is not None: self.actions_stack.append(acts)
        
        return np.asarray(self.states_stack), np.asarray(self.actions_stack)
    
    def get(self):
        return np.asarray(self.states_stack), np.asarray(self.actions_stack)
    
    def get_last_state(self):
        return self.states_stack[-1]

    def get_last_action(self):
        return self.actions_stack[-1]
    
    def append(self, obs=None, acts=None):
        if obs is not None: self.states_stack.append(obs)
        if acts is not None: self.actions_stack.append(acts)

    def reset_stacks(self, init_st=None, init_ac=None):
        if init_ac is None: init_ac = np.zeros(self.action_shape)
        if init_st is None: init_st = np.zeros(self.state_shape) 
        self.actions_stack =   deque(self.n * [init_ac], maxlen=self.n)
        self.states_stack  =   deque(self.n * [init_st], maxlen=self.n)

        return np.asarray(self.states_stack), np.asarray(self.actions_stack)
    
    def fill_with_stack(self, st_stack=None, ac_stack=None):
        if st_stack is not None: self.states_stack = deque(st_stack, maxlen=self.n)
        if ac_stack is not None: self.actions_stack = deque(ac_stack, maxlen=self.n)


    
def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], next_obs=[], delta_obs=[])


# TODO: Hacer una prueba de ablacion para ver si mejora el resultado next_obcuando no se toma
#       En cuenta el estado inicial en el dataset de entrenamiento (cuando el stack no
#       esa lleno)


# TODO: No proveer la accion actual y el siguiente estado en la misma tupla
#       Corregir esto en la linea 45