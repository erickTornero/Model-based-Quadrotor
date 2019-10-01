from wrapper_quad.wrapper_vrep import VREPQuad
import numpy as np

class QuadrotorEnv(VREPQuad):
    def __init__(self, port, reward_type):
        super(QuadrotorEnv, self).__init__(port=port)

        self.faultmotor =   1      
        self.mask       =   np.ones(4, dtype=np.float32)

        if self.faultmotor is not None:
            self.mask[self.faultmotor]  =   0.0

        if reward_type  ==  'type1':
            self.reward =   self.distance_reward
        elif reward_type == 'type2':
            self.reward =   self.roll_pitch_vel_penalized
        elif reward_type == 'type3':
            self.reward =   self.roll_pitch_yaw_vel_penalized
        else:
            print('Error: No valid reward function: example: ("type1")')
    
    def step(self, action:np.ndarray):
        fault_action    =   self.mask * action
        return super(QuadrotorEnv, self).step(fault_action)

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
        yaw_vel             =   next_obs[:, 17]/10.0    
        rel_distance        =   next_obs[:, 9:12]

        #rwdistance          =   np.sqrt(np.sum(rel_distance * rel_distance, axis=1)+yaw_vel*yaw_vel)
        rwdistance          =   np.sqrt(np.sum(rel_distance * rel_distance, axis=1))
        rwdistance          =   4.0 - 1.25 * rwdistance

        vel_penalization    =   np.sqrt(np.sum(roll_pitch_ang_vel * roll_pitch_ang_vel, axis=1)+yaw_vel*yaw_vel)

        return 1.5*rwdistance - vel_penalization
    
    def roll_pitch_yaw_angle_penalized(self, next_obs):
        """
            Hardcode to use nstack = 4
        """
        index_SyawCroll     =   3
        index_Sroll         =   6
        index_SpitchCroll   =   7

        roll_rad            =   np.arcsin(-next_obs[:, index_Sroll])
        cosroll             =   np.cos(roll_rad)
        pitch_rad           =   np.arcsin(next_obs[:, index_SpitchCroll])/cosroll
        yaw_rad             =   np.arcsin(next_obs[:, index_SyawCroll])/cosroll

        ang_pen             =   np.sqrt(roll_rad*roll_rad + pitch_rad*pitch_rad + yaw_rad*yaw_rad)

        reward_angle        =   1.5 - ang_pen
        reward_distance     =   self.distance_reward(next_obs)

        return reward_distance + reward_angle


    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos