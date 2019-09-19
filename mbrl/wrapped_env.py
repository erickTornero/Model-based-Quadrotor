from wrapper_quad.wrapper_vrep import VREPQuad
import numpy as np

class QuadrotorEnv(VREPQuad):
    def __init__(self, port):
        super(QuadrotorEnv, self).__init__(port=port)

    def reward(self, next_obs):
        targetpos   =   self.targetpos
        targetpos   =   np.array([targetpos] * next_obs.shape[0])

        # TODO: Must change if observation space changes
        currpos     =   next_obs[:, 9,12]

        distance    =   targetpos - currpos
        distance    =   np.sqrt(np.sum(distance * distance, axis=1))

        reward      =   4.0 - 1.25 * distance

        return  reward
