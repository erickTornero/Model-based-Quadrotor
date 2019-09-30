from wrapper_quad.wrapper_vrep import VREPQuad
import numpy as np

class QuadrotorEnv(VREPQuad):
    def __init__(self, port, reward_type):
        super(QuadrotorEnv, self).__init__(port=port)

        if reward_type  ==  'type1':
            self.reward =   self.distance_reward
        elif reward_type == 'type2':
            self.reward =   self.roll_pitch_vel_penalized
        elif reward_type == 'type3':
            self.reward =   self.roll_pitch_yaw_vel_penalized
        else:
            print('Error: No valid reward function: example: ("type1")')
    
    def distance_reward(self, next_obs):
        targetpos   =   self.targetpos
        targetpos   =   np.array([targetpos] * next_obs.shape[0])

        # TODO: Must change if observation space changes
        currpos     =   next_obs[:, 9:12]

        #distance    =   targetpos - currpos
        distance    =   currpos
        distance    =   np.sqrt(np.sum(distance * distance, axis=1))

        reward      =   4.0 - 1.25 * distance

        #print(reward)
        return  reward

    def roll_pitch_vel_penalized(self, next_obs):
        # rangues varing from [-10.0 to 10.0] we devide between 10 to scale
        roll_pitch_ang_vel  =   next_obs[:, 15:17]/10.0
        rel_distance        =   next_obs[:, 9:12]

        rwdistance          =   np.sqrt(np.sum(rel_distance * rel_distance, axis=1))
        rwdistance          =   4.0 - 1.25 * rwdistance

        vel_penalization    =   np.sqrt(np.sum(roll_pitch_ang_vel * roll_pitch_ang_vel, axis=1))

        return rwdistance - vel_penalization

    def roll_pitch_yaw_vel_penalized(self, next_obs):
        # rangues varing from [-10.0 to 10.0] we devide between 10 to scale
        roll_pitch_ang_vel  =   next_obs[:, 15:17]/10.0
        yaw_vel             =   next_obs[:, 17]/30.0    
        rel_distance        =   next_obs[:, 9:12]

        rwdistance          =   np.sqrt(np.sum(rel_distance * rel_distance, axis=1)+yaw_vel*yaw_vel)
        rwdistance          =   4.0 - 1.25 * rwdistance

        vel_penalization    =   np.sqrt(np.sum(roll_pitch_ang_vel * roll_pitch_ang_vel, axis=1))

        return rwdistance - vel_penalization

    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos