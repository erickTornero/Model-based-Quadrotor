
import numpy as np
from collections import deque
import itertools

class Runner:
    """
        Collect Samples of quadrotor
    """

    def __init__(self, vecenv, agent, net, mpc, nsteps, n_rollouts):
        self.vec_env    =   vecenv
        self.agent  =   agent
        self.net    =   net
        self.nstack =   net.stack_n

        self.max_path_len =   nsteps
        self.total_samples  =   n_rollouts * nsteps

        self.n_parallel =   agent.n_parallel

        self.env_   =   self.vec_env.getenv


    def run(self, random=False):
        states, actions, rewards, dones = [], [], [], []
        
        paths       =   []
        n_samples   =   0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.n_parallel)]

        # Reset environments
        #obses   =   np.asarray(self.vec_env.reset())
        obses   =   self.vec_env.reset()
        stack_as    =   [StackStAct(self.env_.action_space.shape, self.env_.observation_space.shape, n=4, init_st=ob) for ob in obses]

        while n_samples < self.total_samples:
            if random:
                actions = np.stack([self.env_.action_space.sample() for _ in range(self.n_parallel)], axis=0)
            else:
                # Get next action given stat of actions and states
                obs_stack, act_stack = stack_as.get()
                #actions = mpc.get_actions(obs_stack, act_stack[1:])

            next_obs, rewards, dones, env_infos = self.vec_env.step(actions)

            _   = [stack_.append(obs=next_ob, acts=act) for next_ob, act, stack_ in zip(next_obs, actions, stack_as)]
            # append new samples:


            for idx, stack_, reward, done in zip(itertools.count(), stack_as, rewards, dones):
                observation, action =   stack_.get()
                running_paths[idx]['observations'].append(observation)
                running_paths[idx]['actions'].append(action)
                running_paths[idx]['rewards'].append(reward)
                running_paths[idx]['dones'].append(done)



        
class StackStAct:
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
    
    def append(self, obs=None, acts=None):
        if obs is not None: self.states_stack.append(obs)
        if acts is not None: self.actions_stack.append(acts)

    def reset_stacks(self, init_st=None, init_ac=None):
        if init_ac is None: init_ac = np.zeros(self.action_shape)
        if init_st is None: init_st = np.zeros(self.state_shape) 
        self.actions_stack =   deque(self.n * [init_ac], maxlen=self.n)
        self.states_stack  =   deque(self.n * [init_st], maxlen=self.n)

        return np.asarray(self.states_stack), np.asarray(self.actions_stack)

    
def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[])


# TODO: Hacer una prueba de ablacion para ver si mejora el resultado cuando no se toma
#       En cuenta el estado inicial en el dataset de entrenamiento (cuando el stack no
#       esa lleno)


# TODO: No proveer la accion actual y el siguiente estado en la misma tupla
#       Corregir esto en la linea 45