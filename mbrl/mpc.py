# Python file to create Model Predictive Controllers

import numpy as np
import gym
class RandomShooter:
    def __init__(self, h, c, env_:gym.Env, discount=1.0):
        self.horizon    =   h
        self.candidates =   c
        self.env        =   env_
        self.discount   =   discount

        self.act_space  =   self.env.action_space
        self.obs_space  =   self.env.observation_space  

    def get_action(self, dynamics, observations):
        h   =   self.horizon
        c   =   self.candidates
        
        actions =   self.get_random_actions(h * c)

    def get_random_actions(self, n):
        return np.random.uniform(low=self.act_space.low, high=self.act_space.high, size=(n,)+self.act_space.shape)