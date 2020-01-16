import gym
import gym_reinmav
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gym_reinmav.envs.mujoco import MujocoQuadEnv
from gym import spaces

class QuadrotorMujocoEnv(gym.Env):
    def __init__(self, port=None, reward_type='type1', fault_rotor=None):
        super(QuadrotorMujocoEnv, self).__init__()
        self.mujocoenv          =   MujocoQuadEnv()
        self.faultmotor         =   fault_rotor
        self.mask               =   np.ones(4, dtype=np.float32)
        self.observation_space  =   spaces.Box(-np.inf, np.inf, (22,), np.float32)
        self.action_space       =   self.mujocoenv.action_space

        if reward_type == 'type8':
            self.reward =   pos_roll_pitch_rot_penalization

    def step(self, action:np.ndarray):
        fault_action                =   self.mask * action
        native_obs, _, _, info  =   self.mujocoenv.step(fault_action)
        
        obs     =   self._transform_observation(native_obs)
        reward  =   self.compute_rewards_distance(obs) 
        done    =   self.compute_done(obs)

        return obs, reward, done, info
    
    def reset(self):
        obs = self.mujocoenv.reset()
        return self._transform_observation(obs)

    def _transform_observation(self, _native_obs):
        """
            Reinmav Quadrotor mujoco, distribution of observation_space
            [qpos, qvel]
            [{pos[0-2], orientation[3-6]}, {lin_speed[7-9], rot_speed[10-12]}]--> (13, )
        """
        
        orient  =   _native_obs[3:7]
        pos     =   _native_obs[0:3]
        lin_vel =   _native_obs[7:10]
        rot_vel =   _native_obs[10:13]

        rot_mat =   Rotation.from_quat(orient).as_matrix().flatten()
        
        return np.concatenate((rot_mat, pos, lin_vel, rot_vel, orient))
    
    def compute_rewards_distance(self, _obs):
        position    =   _obs[9:12]
        magnitud    =   np.sqrt((position * position).sum())

        return 4.0 - 1.25 * magnitud

    def compute_done(self, _obs):
        position    =   _obs[9:12]
        magnitud    =   np.sqrt((position * position).sum())

        done    =   magnitud > 3.2
        return done
    
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
