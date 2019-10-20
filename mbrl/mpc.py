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
        #self.batch_as   =   BatchStacks(self.act_space.shape, self.obs_space.shape, self.stack_n, self.candidates, device)
        self.batch_as   =   BatchStacksTorch(self.act_space.shape, self.obs_space.shape, self.stack_n, self.candidates, device)

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
    
    def get_action_torch(self, obs_):
        obs_np, acts_np =   obs_.get()
        self.batch_as.restart(torch.tensor(obs_np, dtype=torch.float32, device=self.device), torch.tensor(acts_np, dtype=torch.float32, device=self.device))
        h   =   self.horizon
        c   =   self.candidates
        returns =   torch.zeros((c,), dtype=torch.float32, device=self.device)

        #must_change to torch
        actions =   self.get_random_actions_torch(h * c).reshape((h, c) + self.act_space.shape)

        actions =   actions.reshape((h, c, self.act_space.shape[0]))
        for t in range(h):
            if t == 0:
                action_c    =   actions[t]
            
            self.batch_as.slide_action_stack(actions[t])
            obs_flat    =   self.batch_as.get()
            # must change to torch
            obs_flat    =   self.normalize_torch(obs_flat)

            # must
            next_obs    =   self.dynamics.predict_next_obs(obs_flat, self.device)
            # must return in torch
            rewards     =   self.env.reward(next_obs)
            returns     =   returns + self.discount**t*rewards

            self.batch_as.slide_state_stack(next_obs)

        return np.asarray(action_c[torch.argmax(returns).item()].to('cpu'))

    def get_action_CEM(self, obs_, J:int, M:int, alpha:int):
        """
            Planning with Random Shooting with Cross Entropy Method for a single Environment

            obs_:       Observation for planning actions
            M:          Number of interactions
            J:          Number of maximum values to take
        """
        self.batch_as.restart(*obs_.get())
        h   =   self.horizon
        c   =   self.candidates
        returns =   np.zeros((c, ))

        high    =   self.act_space.high
        low     =   self.act_space.low
        
        for m in range(M):
            """ 
                If first iteration: execute the Random Shooting algorithm
                else: Iterate with Cross Entropy Method (CEM)
            """
            if m==0:
                actions =   self.get_random_actions(h * c).reshape((h, c,) + self.act_space.shape)
                actions =   actions.reshape((c, h, self.act_space.shape[0]))

                actions_mean    =   np.zeros((self.horizon, self.act_space.shape[0]), dtype=np.float32)
                actions_std     =   np.zeros((self.horizon, self.act_space.shape[0]), dtype=np.float32)
            else:
                actions         =   np.random.normal(actions_mean, actions_std, (c, h, self.act_space.shape[0]))
                actions         =   np.clip(actions, a_min=low, a_max=high)
            for t in range(h):
               
                self.batch_as.slide_action_stack(actions[:,t,:])
                """ Normalize input """
                obs_flat    =   self.batch_as.get()
                obs_flat    =   self.normalize_(obs_flat)
                obs_tensor  =   torch.tensor(obs_flat, dtype=torch.float32, device=self.device)

                next_obs    =   self.dynamics.predict_next_obs(obs_tensor, self.device).to('cpu')
                next_obs    =   np.asarray(next_obs)
                rewards     =   self.env.reward(next_obs)
                returns     =   returns + self.discount**t*rewards

                self.batch_as.slide_state_stack(next_obs)
            
            best_j_indexes =   np.argsort(returns)[-J:][::-1]
            best_j_actions  =   actions[np.ix_(best_j_indexes)]
            actions_mean    =   alpha * np.mean(best_j_actions, axis=0) + actions_mean * (1 - alpha)
            actions_std     =   alpha * np.std(best_j_actions,  axis=0) + actions_std * (1 - alpha)

        return best_j_actions[0][0]

    def get_action_PDDM(self, obs_, gamma, beta):
        import copy
        self.batch_as.restart(*obs_.get())
        h   =   self.horizon
        c   =   self.candidates
        returns =   np.zeros((c, ))
        #actions =   self.get_random_actions(h * c).reshape((h, c,) + self.act_space.shape)

        actions_mean    =   np.zeros((self.horizon, self.act_space.shape[0]), dtype=np.float32)
        noises          =   np.zeros((self.horizon, c, self.act_space.shape[0]), dtype=np.float32)
        #set_trace()
        #u_ti            =   np.random.normal(, 10.0, size=(c, 4))
        for t in range(h):
            # Compute random actions
            actions =   self.get_random_actions(c).reshape((c,) + self.act_space.shape)

            batch_copied    =   copy.deepcopy(self.batch_as)

            batch_copied.slide_action_stack(actions)
            """ Normalize input """
            obs_flat    =   batch_copied.get()
            obs_flat    =   self.normalize_(obs_flat)
            obs_tensor  =   torch.tensor(obs_flat, dtype=torch.float32, device=self.device)

            next_obs    =   self.dynamics.predict_next_obs(obs_tensor, self.device).to('cpu')
            next_obs    =   np.asarray(next_obs)
            rewards     =   self.env.reward(next_obs)
            #returns     =   returns + self.discount**t*rewards
            """Compute ponderate mean action:"""
            gamrwclamp  =   np.clip(gamma * rewards, -70.0, 70.0)
            actions_mean    =   np.sum(np.exp(gamrwclamp).reshape(-1, 1)* actions, axis=0)/(np.sum(np.exp(gamrwclamp))) 
            """ 
                Compute noises 
                TODO: What is the scale of Covarianze matrix, temporaly we try with 10.0?
            """
            u_ti            =   np.random.normal(np.zeros(self.act_space.shape, dtype=np.float32), 30.0, size=((c,)+self.act_space.shape))
            noises[t]       =   beta * u_ti + ((1 - beta) * noises[t - 1] if t > 0 else 0)

            actions_pddm    =   np.clip(noises[t] + actions_mean, 0.0, 100.0)

            if t == 0:
                action_c   =   actions_pddm

            self.batch_as.slide_action_stack(actions_pddm)
            """ Normalize input """
            obs_flat    =   self.batch_as.get()
            obs_flat    =   self.normalize_(obs_flat)
            obs_tensor  =   torch.tensor(obs_flat, dtype=torch.float32, device=self.device)

            next_obs    =   self.dynamics.predict_next_obs(obs_tensor, self.device).to('cpu')
            next_obs    =   np.asarray(next_obs)
            rewards     =   self.env.reward(next_obs)
            returns     =   returns + self.discount**t*rewards

            self.batch_as.slide_state_stack(next_obs)

        return action_c[np.argmax(returns)]

    def get_random_actions(self, n):
        return np.random.uniform(low=self.act_space.low, high=self.act_space.high, size=(n,)+self.act_space.shape)
    
    def get_random_actions_torch(self, n):
        acts    =   np.random.uniform(low=self.act_space.low, high=self.act_space.high, size=(n,)+self.act_space.shape)
        return torch.tensor(acts, dtype=torch.float32, device=self.device)

    def normalize_(self, obs):
        assert self.dynamics.mean_input is not None
        return (obs - self.dynamics.mean_input)/(self.dynamics.std_input + self.dynamics.epsilon)
    def normalize_torch(self, obs):
        assert self.dynamics.mean_input is not None
        return (obs - torch.tensor(self.dynamics.mean_input, dtype=torch.float32, device=self.device))/(torch.tensor(self.dynamics.std_input, dtype=torch.float32, device=self.device)+self.dynamics.epsilon)

    def denormalize_(self, obs):
        assert self.dynamics.mean_input is not None
        return obs * (self.dynamics.std_input + self.dynamics.epsilon) + self.dynamics.mean_input

class CrossEntropyMethod:
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

class MPPI:
    def __init__(self, h, c, env_:gym.Env, dynamics, device, discount):
        from collections import deque
        self.horizon    =   h
        self.candidates =   c
        self.env        =   env_
        self.discount   =   discount

        self.act_space  =   self.env.action_space
        self.obs_space  =   self.env.observation_space 

        self.dynamics   =   dynamics
        self.device     =   device

        self.initial_sequence   =   deque(self.horizon * [np.zeros(self.act_space.shape[0], dtype=np.float32)], maxlen=self.horizon)

        self.stack_n    =   self.dynamics.stack_n
        self.batch_as   =   BatchStacks(self.act_space.shape, self.obs_space.shape, self.stack_n, self.candidates, device)

    def get_action(self, obs_):
        self.batch_as.restart(*obs_.get())
        h   =   self.horizon
        c   =   self.candidates
        returns =   np.zeros((c, ))
        init_sequence   =   np.asarray(self.initial_sequence)



            
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
        @parameters:

        act_shape      :   Action space shape
        st_shape       :   State space shape
        stack_n        :   Stacked state-actions-pairs (usually: 4)
        n              :   Number of candidates
        device         :   Pytorch variable, to compute in cpu or gpu
        init_st_stack  :   Initial state_stack
        init_ac_stack  :   Initial actions stack

        restart function must be called to used properly
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


class BatchStacksTorch:
    """
        Append a batch of state-actions: (StackStAct)get
        Optimized, working with *torch.Tensor* data-type
        ans with = are not really copy, just share memory
        @parameters:

        act_shape      :   Action space shape
        st_shape       :   State space shape
        stack_n        :   Stacked state-actions-pairs (usually: 4)
        n              :   Number of candidates
        device         :   Pytorch variable, to compute in cpu or gpu
        init_st_stack  :   Initial state_stack
        init_ac_stack  :   Initial actions stack

        restart function must be called to used properly
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
        #Assuming torch device
        if init_st_stack is not None:
            self.state_batch_flat   =   init_st_stack.flatten()
            self.state_batch_flat   =   self.state_batch_flat.repeat(n, 1)
            """Ensure compatibilities of shapes"""
            assert self.state_batch_flat.shape[1]  == st_shape[0] * stack_n
        
        if init_ac_stack is not None:
            self.action_batch_flat  =   init_ac_stack.flatten()
            self.action_batch_flat  =   self.action_batch_flat.repeat(n, 1)
            """Ensure compatibilities of shapes"""
            assert self.action_batch_flat.shape[1]  ==  act_shape[0] * stack_n

    def restart(self, init_st_stack, init_ac_stack):
        self.state_batch_flat   =   init_st_stack.flatten()
        self.state_batch_flat   =   self.state_batch_flat.repeat(self.n, 1)

        self.action_batch_flat  =   init_ac_stack.flatten()
        self.action_batch_flat  =   self.action_batch_flat.repeat(self.n, 1)
        """Ensure compatibilities of shapes"""
        assert self.state_batch_flat.shape[1]  ==   self.st_shape[0] * self.stack_n
        assert self.action_batch_flat.shape[1]  ==  self.act_shape[0] * self.stack_n


    def slide_action_stack(self, entry_action):
        self.action_batch_flat[:, :self.action_shape_sz * (self.stack_n - 1)]   =   self.action_batch_flat[:, self.action_shape_sz:]
        self.action_batch_flat[:, self.action_shape_sz * (self.stack_n - 1):]   =   entry_action

    def slide_state_stack(self, entry_state):
        self.state_batch_flat[:, :self.state_shape_sz * (self.stack_n - 1)]     = self.state_batch_flat[:, self.state_shape_sz:]
        self.state_batch_flat[:, self.state_shape_sz * (self.stack_n - 1):]     = entry_state
    
    def slide_stacks(self, entry_action=None, entry_state=None):
        if entry_action is not None: self.slide_action_stack(entry_action)
        if entry_state is not None: self.slide_state_stack(entry_state)
    
    def get(self):
        #return self.state_batch_flat, self.action_batch_flat
        return torch.cat((self.state_batch_flat, self.action_batch_flat), axis=1)
    def get_tensor_numpy(self):
        np_obs = self.get().to('cpu')
        #return torch.from_numpy(np_obs).to(self.device)
        return np.asarray(np_obs)