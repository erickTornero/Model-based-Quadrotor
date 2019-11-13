from wrapper_quad.wrapper_vrep import VREPQuad
from wrapper_quad.wrapper_vrep2 import VREPQuadAccel
from wrapper_quad.wrapper_vrep3 import VREPQuadSimple
from wrapper_quad.wrapper_q1 import VREPQuadAccelRot, VREPQuadRotmat, VREPQuadRotmatAugment
import numpy as np
import torch

class QuadrotorEnv(VREPQuadRotmat):
    def __init__(self, port, reward_type, fault_rotor=None):
        super(QuadrotorEnv, self).__init__(port=port)

        self.faultmotor =   fault_rotor
        self.mask       =   np.ones(4, dtype=np.float32)
                
        if self.faultmotor is not None:
            assert fault_rotor < 4, 'Choose a fault rotor in range of [0-3]'
            self.mask[self.faultmotor]  =   0.0
            print('{} Initialized with rotor {} faulted, and reward: {}'.format(self.__class__.__name__, self.faultmotor, reward_type))
        else: print('{} Initialized in fault-free case, and reward: {}'.format(self.__class__.__name__, reward_type))
        
        """ Initialize Reward function """
        if reward_type  ==  'type1':
            self.reward =   self.distance_reward_torch
        elif reward_type == 'type2':
            self.reward =   self.roll_pitch_vel_penalized
        elif reward_type == 'type3':
            self.reward =   self.roll_pitch_yaw_vel_penalized
        elif reward_type == 'type4':
            self.reward = self.roll_pitch_angle_penalized
        else:
            assert True, 'Error: No valid reward function: example: ("type1")'
    
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
    def distance_reward_torch(self, next_obs):
        currpos =   next_obs[:, 9:12]

        distance    =   torch.sqrt(torch.sum(currpos * currpos, dim=1))

        reward      =   4.0 - 1.25 * distance
        return reward

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
            statespaceof 18: (rotmat, pos, lin_vel, ang_vel)
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

    def roll_pitch_angle_penalized(self, next_obs):
        """
            Hardcode to use nstack = 4
            statespaceof 18: (rotmat, pos, lin_vel, ang_vel)
        """
        #index_SyawCroll     =   3
        index_Sroll         =   6
        index_SpitchCroll   =   7

        roll_sin            =   -next_obs[:, index_Sroll]
        roll_rad            =   torch.asin(roll_sin)
        cosroll             =   torch.cos(roll_rad)
        pitch_sin           =   next_obs[:, index_SpitchCroll]/(cosroll + 1e-5)
        #yaw_rad             =   next_obs[:, index_SyawCroll]/(cosroll + 1e-5)

        ang_pen             =   roll_sin*roll_sin + pitch_sin*pitch_sin

        reward_angle        =   2.0 - ang_pen
        reward_distance     =   self.distance_reward_torch(next_obs)

        return reward_distance + reward_angle


    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos


class QuadrotorEnvAugment(VREPQuadRotmatAugment):
    def __init__(self, port, reward_type, fault_rotor=None):
        super(QuadrotorEnvAugment, self).__init__(port=port)

        self.faultmotor =   fault_rotor
        self.mask       =   np.ones(4, dtype=np.float32)
                
        if self.faultmotor is not None:
            assert fault_rotor < 4, 'Choose a fault rotor in range of [0-3]'
            self.mask[self.faultmotor]  =   0.0
            print('{} Initialized with rotor {} faulted, and reward: {}'.format(self.__class__.__name__, self.faultmotor, reward_type))
        else: print('{} Initialized in fault-free case, and reward: {}'.format(self.__class__.__name__, reward_type))
        
        """ Initialize Reward function """
        if reward_type  ==  'type1':
            self.reward =   self.distance_reward_torch
        elif reward_type == 'type2':
            self.reward =   self.roll_pitch_vel_penalized
        elif reward_type == 'type3':
            self.reward =   self.roll_pitch_yaw_vel_penalized
        elif reward_type == 'type4':
            self.reward = self.roll_pitch_angle_penalized
        elif reward_type == 'type5':
            self.reward = self.roll_pitch_angle_rotyaw_penalized
        else:
            assert True, 'Error: No valid reward function: example: ("type1")'
    
    def step(self, action:np.ndarray):
        fault_action    =   self.mask * action
        return super(QuadrotorEnvAugment, self).step(fault_action)

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
    def distance_reward_torch(self, next_obs):
        currpos =   next_obs[:, 9:12]

        distance    =   torch.sqrt(torch.sum(currpos * currpos, dim=1))

        reward      =   4.0 - 1.25 * distance
        return reward

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
            statespaceof 18: (rotmat, pos, lin_vel, ang_vel)
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

    def roll_pitch_angle_penalized(self, next_obs):
        """
            reward_type:    'type4'
            Hardcoded
            statespaceof 18: (rotmat, pos, lin_vel, ang_vel, orientation)
        """
        #index_SyawCroll     =   3
        index_roll          =   18
        index_pitch         =   19

        roll_rad            =   next_obs[:, index_roll]
        
        pitch_rad           =   next_obs[:, index_pitch]
        #yaw_rad             =   next_obs[:, index_SyawCroll]/(cosroll + 1e-5)

        ang_pen             =   roll_rad*roll_rad + pitch_rad*pitch_rad

        reward_angle        =   2.0 - ang_pen
        reward_distance     =   self.distance_reward_torch(next_obs)

        return reward_distance + reward_angle

    def roll_pitch_angle_rotyaw_penalized(self, next_obs):
        """
            reward_type:    'type5'
            Hardcoded
            statespaceof 18: (rotmat, pos, lin_vel, ang_vel, orientation)
        """
        #index_SyawCroll     =   3
        index_roll          =   18
        index_pitch         =   19

        index_rot_yaw       =   17

        """ Reward angular """
        roll_rad            =   next_obs[:, index_roll]
        
        pitch_rad           =   next_obs[:, index_pitch]
        #yaw_rad             =   next_obs[:, index_SyawCroll]/(cosroll + 1e-5)

        ang_pen             =   roll_rad*roll_rad + pitch_rad*pitch_rad

        reward_angle        =   2.0 - ang_pen
        """ 
            Reward angular rotation Yaw axis 
            The definition of the scaled reward is based on distribution of angular speed
            utils/analize_paths/plot_ang_velocity.py
        """
        yaw_speed           =   next_obs[:, index_rot_yaw]

        reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(200.0)
        #reward_yaw_speed    =   2.0 - (yaw_speed * yaw_speed)/100.0

        """ distance reward """
        reward_distance     =   self.distance_reward_torch(next_obs)

        return reward_distance + reward_angle + reward_yaw_speed

    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos
    

class QuadrotorAcelEnv(VREPQuadAccel):
    def __init__(self, port, reward_type, fault_rotor=None):
        super(QuadrotorAcelEnv, self).__init__(port=port)
        
        self.faultmotor =   fault_rotor
        self.mask       =   np.ones(4, dtype=np.float32)
                
        if self.faultmotor is not None:
            assert fault_rotor < 4, 'Choose a fault rotor in range of [0-3]'
            self.mask[self.faultmotor]  =   0.0
            print('QuadrotorAcelEnv Initialized with rotor {} faulted, and reward: {}'.format(self.faultmotor, reward_type))
        else: print('QuadrotorAcelEnv Initialized in fault-free case, and reward: {}'.format(reward_type))
        
        """ Initialize Reward function """
        if reward_type  ==  'type1':
            self.reward =   self.distance_reward
        else:
            assert True, 'Error: No valid reward function: example: ("type1")'
        

    def step(self, action:np.ndarray):
        fault_action    =   self.mask * action
        return super(QuadrotorAcelEnv, self).step(fault_action)

    def distance_reward(self, next_obs):
        currpos =   next_obs[:, 0:3]

        distance    =   torch.sqrt(torch.sum(currpos * currpos, dim=1))

        reward      =   4.0 - 1.25 * distance
        return reward

    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos


class QuadrotorSimpleEnv(VREPQuadSimple):
    def __init__(self, port, reward_type, fault_rotor=None):
        super(QuadrotorSimpleEnv, self).__init__(port=port)
        
        self.faultmotor =   fault_rotor
        self.mask       =   np.ones(4, dtype=np.float32)
                
        if self.faultmotor is not None:
            assert fault_rotor < 4, 'Choose a fault rotor in range of [0-3]'
            self.mask[self.faultmotor]  =   0.0
            print('QuadrotorSimpleEnv Initialized with rotor {} faulted, and reward: {}'.format(self.faultmotor, reward_type))
        else: print('QuadrotorSimpleEnv Initialized in fault-free case, and reward: {}'.format(reward_type))
        
        """ Initialize Reward function """
        if reward_type  ==  'type1':
            self.reward =   self.distance_reward
        else:
            assert True, 'Error: No valid reward function: example: ("type1")'
        

    def step(self, action:np.ndarray):
        fault_action    =   self.mask * action
        return super(QuadrotorSimpleEnv, self).step(fault_action)

    def distance_reward(self, next_obs):
        currpos =   next_obs[:, 0:3]

        distance    =   torch.sqrt(torch.sum(currpos * currpos, dim=1))

        reward      =   4.0 - 1.25 * distance
        return reward
    
    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos

class QuadrotorAcelRotmat(VREPQuadAccelRot):
    def __init__(self, port, reward_type, fault_rotor=None):
        super(QuadrotorAcelRotmat, self).__init__(port=port)
        self.faultmotor =   fault_rotor
        self.mask       =   np.ones(4, dtype=np.float32)

        if self.faultmotor is not None:
            assert fault_rotor < 4, 'Choose a fault rotor in range of [0-3]'
            self.mask[self.faultmotor]  =   0.0
            print('QuadrotorAcelRotmat Initialized with rotor {} faulted, and reward: {}'.format(self.faultmotor, reward_type))
        else: print('QuadrotorAcelRotmat Initialized in fault-free case, and reward: {}'.format(reward_type))
        
        """ Initialize Reward function """
        if reward_type  ==  'type1':
            self.reward =   self.distance_reward
        else:
            assert True, 'Error: No valid reward function: example: ("type1")'
    
    def step(self, action:np.ndarray):
        fault_action    =   self.mask * action
        return super(QuadrotorAcelRotmat, self).step(fault_action)

    def distance_reward(self, next_obs):
        currpos =   next_obs[:, 0:3]

        distance    =   torch.sqrt(torch.sum(currpos * currpos, dim=1))

        reward      =   4.0 - 1.25 * distance
        return reward

    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos


