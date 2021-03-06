from wrapper_quad.wrapper_vrep import VREPQuad
from wrapper_quad.wrapper_vrep2 import VREPQuadAccel
from wrapper_quad.wrapper_vrep3 import VREPQuadSimple
from wrapper_quad.wrapper_q1 import VREPQuadAccelRot, VREPQuadRotmat, VREPQuadRotmatAugment, VREPQuadQuaternionAugment
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
    def distance_reward_torch(self, next_obs, actions=None):
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
        elif reward_type == 'type6':
            self.reward = self.roll_pitch_angle_rotyaw_input_penalized
        elif reward_type == 'type7':
            self.reward = self.pos_rot_penalization
        elif reward_type == 'type8':
            self.reward = self.pos_roll_pitch_rot_penalization
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

        #reward_angle        =   2.0 - ang_pen
        #reward_angle        =   - ang_pen/2.0
        reward_angle        =   - ang_pen/8.0
        #reward_angle        =   - ang_pen/16.0
        """ 
            Reward angular rotation Yaw axis 
            The definition of the scaled reward is based on distribution of angular speed
            utils/analize_paths/plot_ang_velocity.py
        """
        yaw_speed           =   next_obs[:, index_rot_yaw]

        #reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(200.0)
        #reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(400.0)
        reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(1000.0)
        #reward_yaw_speed    =   - torch.abs(-15.0-yaw_speed)/5.0        
        #reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(1500.0)
        #reward_yaw_speed    =   2.0 - (yaw_speed * yaw_speed)/100.0

        """ distance reward """
        reward_distance     =   self.distance_reward_torch(next_obs)

        return reward_distance + reward_angle + reward_yaw_speed
    
    def roll_pitch_angle_rotyaw_input_penalized(self, next_obs, acts):
        """ 
            reward_type 6
            Penalization of inputs [0-0.5] of penalization
        """ 

        rp      =   self.roll_pitch_angle_rotyaw_penalized(next_obs)
        act_pen =   -5e-5*torch.sum(acts * acts, dim=1)
        return rp + act_pen
    
    def pos_rot_penalization(self, next_obs, acts):
        """
            reward_type 7
            &   Position penalization
            &   rotation speed penalization, in 3 axes
        """
        index_start_rot         =   15
        #constant_roll_pitch_rot =   -1e-1
        #constant_yaw_rot        =   -1e-2
        constant_roll_pitch_rot =   -1e-1
        constant_yaw_rot        =   -1e-2
        reward_distance         =   2 * self.distance_reward_torch(next_obs)
        rotation_speeds         =   next_obs[:, index_start_rot:index_start_rot + 3]
        rotation_speeds         =   rotation_speeds * rotation_speeds
        reward_speeds           =   constant_roll_pitch_rot * torch.sum(rotation_speeds[:,:2], axis=1) + constant_yaw_rot * rotation_speeds[:,-1]

        return reward_distance + reward_speeds
    
    def pos_roll_pitch_rot_penalization(self, next_obs, acts):
        """ 
            reward_type 8
            &   Penalization of position,
            &   rotation roll-pitch penalization
        """
        index_start_rot         =   15
        constant_roll_pitch_rot =   -1e-2
        constant_yaw_rot        =   -1e-3
        #constant_roll_pitch_rot =   -1e-1
        #constant_yaw_rot        =   0.0
        reward_distance         =   2 * self.distance_reward_torch(next_obs)
        rotation_speeds         =   next_obs[:, index_start_rot:index_start_rot + 3]
        rotation_speeds         =   rotation_speeds * rotation_speeds
        reward_speeds           =   constant_roll_pitch_rot * torch.sum(rotation_speeds[:,:2], axis=1) + constant_yaw_rot * rotation_speeds[:,-1]

        return reward_distance + reward_speeds

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


class QuadrotorQuaternionAugment(VREPQuadQuaternionAugment):
    def __init__(self, port, reward_type, fault_rotor=None):
        super(QuadrotorQuaternionAugment, self).__init__(port=port)
        
        self.faultmotor =   fault_rotor
        self.mask       =   np.ones(4, dtype=np.float32)
                
        if self.faultmotor is not None:
            assert fault_rotor < 4, 'Choose a fault rotor in range of [0-3]'
            self.mask[self.faultmotor]  =   0.0
            print('{} Initialized with rotor {} faulted, and reward: {}'.format(self.__class__.__name__, self.faultmotor, reward_type))
        else: print('{} Initialized in fault-free case, and reward: {}'.format(self.__class__.__name__, reward_type))

        """ Initialize Reward function """
        if reward_type  ==  'type1':
            self.reward =   None
        elif reward_type == 'type2':
            self.reward =   None
        elif reward_type == 'type3':
            self.reward =   None
        elif reward_type == 'type4':
            self.reward = None
        elif reward_type == 'type5':
            self.reward = self.roll_pitch_angle_rotyaw_penalized
        else:
            assert True, 'Error: No valid reward function: example: ("type1")'
        
    def step(self, action:np.ndarray):
        fault_action    =   self.mask * action
        return super(QuadrotorQuaternionAugment, self).step(fault_action)
    
    def distance_reward_torch(self, next_obs):
        currpos =   next_obs[:, 0:3]
        distance    =   torch.sqrt(torch.sum(currpos * currpos, dim=1))

        reward      =   4.0 - 1.25 * distance
        return reward
    def roll_pitch_angle_rotyaw_penalized(self, next_obs):
        """
            reward_type:    'type5'
            Hardcoded
            statespaceof 18: (rotmat, pos, lin_vel, ang_vel, orientation)
        """
        #index_SyawCroll     =   3
        index_roll          =   13
        index_pitch         =   14

        index_rot_yaw       =   8

        """ Reward angular """
        roll_rad            =   next_obs[:, index_roll]
        
        pitch_rad           =   next_obs[:, index_pitch]
        #yaw_rad             =   next_obs[:, index_SyawCroll]/(cosroll + 1e-5)

        ang_pen             =   roll_rad*roll_rad + pitch_rad*pitch_rad

        #reward_angle        =   2.0 - ang_pen
        #reward_angle        =   - ang_pen/2.0
        reward_angle        =   - ang_pen/8.0
        #reward_angle        =   - ang_pen/16.0
        """ 
            Reward angular rotation Yaw axis 
            The definition of the scaled reward is based on distribution of angular speed
            utils/analize_paths/plot_ang_velocity.py
        """
        yaw_speed           =   next_obs[:, index_rot_yaw]

        #reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(200.0)
        #reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(400.0)
        reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(1000.0)
        #reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(1500.0)
        #reward_yaw_speed    =   2.0 - (yaw_speed * yaw_speed)/100.0

        """ distance reward """
        reward_distance     =   self.distance_reward_torch(next_obs)

        return reward_distance + reward_angle + reward_yaw_speed

    def set_targetpos(self, tpos:np.ndarray):
        assert tpos.shape[0]    ==  3
        self.targetpos  =   tpos
