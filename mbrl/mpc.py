# Python file to create Model Predictive Controllers

import numpy as np
import gym
import torch

from IPython.core.debugger import set_trace


class RandomShooter:
    def __init__(self, h, c, env_:gym.Env, dynamics, device, discount=1.0):
        self.horizon    =   h
        self.candidates =   c
        self.env        =   env_
        self.discount   =   discount

        self.act_space  =   self.env.action_space
        self.obs_space  =   self.env.observation_space 

        self.dynamics   =   dynamics
        self.device     =   device

        self.stack_n    =   self.dynamics.stack_n
        self.batch_as   =   BatchStacks(self.act_space.shape, self.obs_space.shape, self.stack_n, self.candidates, device)

    def get_action(self, obs_):
        """
            Planning with Random Shooting for a single Environment
            dynamics:   Dynamics given by a nn
            obs_:       Observation for planning actions
        """
        self.batch_as.restart(*obs_.get())

        h   =   self.horizon
        c   =   self.candidates
        returns =   np.zeros((c, ))
        #set_trace()
        actions =   self.get_random_actions(h * c).reshape((h, c,) + self.act_space.shape)

        actions =   actions.reshape((h, c, self.act_space.shape[0]))

        for t in range(h):
            if t == 0:
                action_c    =   actions[t]

            self.batch_as.slide_action_stack(actions[t])
            """ Normalize input """
            obs_flat    =   self.batch_as.get()
            obs_flat    =   self.normalize_(obs_flat)
            obs_tensor  =   torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
            
            next_obs    =   self.dynamics.predict_next_obs(obs_tensor, self.device).to('cpu')
            next_obs    =   np.asarray(next_obs)
            rewards     =   self.env.reward(next_obs)
            returns     =   returns + self.discount**t*rewards

            self.batch_as.slide_state_stack(next_obs)
        #idx_max_ret =   np.argmax(returns)
        #idx_min_ret =   np.argmin(returns)
        #print(returns[idx_min_ret], returns[idx_max_ret])
        return action_c[np.argmax(returns)]


    def get_random_actions(self, n):
        return np.random.uniform(low=self.act_space.low, high=self.act_space.high, size=(n,)+self.act_space.shape)
    
    def normalize_(self, obs):
        assert self.dynamics.mean_input is not None
        return (obs - self.dynamics.mean_input)/(self.dynamics.std_input + self.dynamics.epsilon)
    def denormalize_(self, obs):
        assert self.dynamics.mean_input is not None
        return obs * (self.dynamics.std_input + self.dynamics.epsilon) + self.dynamics.mean_input
        

class BatchStacks:
    """
        Append a batch of state-actions: (StackStAct)get
        Optimized, working with np.ndarray data-type
        ans with = are not really copy, just share memory
    """
    def __init__(self, act_shape, st_shape, stack_n, n:int, device, init_st_stack=None, init_ac_stack=None):
        #b_stack =   StackStAct(act_shape, st_shape, stack_n, init_st, init_ac)
        self.state_init =   0
        self.action_init    =   st_shape[0] * stack_n

        self.action_shape_sz    =   act_shape[0]
        self.state_shape_sz     =   st_shape[0]
        self.stack_n            =   stack_n

        self.n                  =   n
        self.act_shape          =   act_shape
        self.st_shape           =   st_shape
        self.device             =   device
        #set_trace()
        if init_st_stack is not None:
            self.state_batch_flat   =   init_st_stack.flatten()
            self.state_batch_flat   =   np.tile(self.state_batch_flat, (n, 1))
            """Ensure compatibilities of shapes"""
            assert self.state_batch_flat.shape[1]  == st_shape[0] * stack_n
        
        if init_ac_stack is not None:
            self.action_batch_flat  =   init_ac_stack.flatten()
            self.action_batch_flat   =   np.tile(self.action_batch_flat, (n, 1))
            """Ensure compatibilities of shapes"""
            assert self.action_batch_flat.shape[1]  ==  act_shape[0] * stack_n

    def restart(self, init_st_stack, init_ac_stack):
        self.state_batch_flat   =   init_st_stack.flatten()
        self.state_batch_flat   =   np.tile(self.state_batch_flat, (self.n, 1))

        self.action_batch_flat  =   init_ac_stack.flatten()
        self.action_batch_flat   =   np.tile(self.action_batch_flat, (self.n, 1))
        """Ensure compatibilities of shapes"""
        assert self.state_batch_flat.shape[1]  ==   self.st_shape[0] * self.stack_n
        assert self.action_batch_flat.shape[1]  ==  self.act_shape[0] * self.stack_n


    def slide_action_stack(self, entry_action):
        self.action_batch_flat[:, :self.action_shape_sz * (self.stack_n - 1)]   =   self.action_batch_flat[:, self.action_shape_sz:]
        self.action_batch_flat[:, self.action_shape_sz * (self.stack_n - 1):]    =   entry_action

    def slide_state_stack(self, entry_state):
        self.state_batch_flat[:, :self.state_shape_sz * (self.stack_n - 1)] = self.state_batch_flat[:, self.state_shape_sz:]
        self.state_batch_flat[:, self.state_shape_sz * (self.stack_n - 1):] = entry_state
    
    def slide_stacks(self, entry_action=None, entry_state=None):
        if entry_action is not None: self.slide_action_stack(entry_action)
        if entry_state is not None: self.slide_state_stack(entry_state)
    
    def get(self):
        #return self.state_batch_flat, self.action_batch_flat
        return np.concatenate((self.state_batch_flat, self.action_batch_flat), axis=1)
    def get_tensor_torch(self):
        np_obs = self.get()
        #return torch.from_numpy(np_obs).to(self.device)
        return torch.tensor(np_obs, dtype=torch.float32, device=self.device)