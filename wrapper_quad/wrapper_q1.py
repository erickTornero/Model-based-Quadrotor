
from wrapper_quad.wrapper import WrapperQuad
import wrapper_quad.vrep as vrep
from wrapper_quad.utility import GetFlatRotationMatrix
from collections import OrderedDict
from gym import spaces
import numpy as np

class VREPQuadAccelRot(WrapperQuad):
    def __init__(self, ip='127.0.0.1', port=19997):
        super(VREPQuadAccelRot, self).__init__(ip=ip, port=port)

        self.action_space       =   spaces.Box(low=0.0,high=100.0,shape=(4,), dtype=np.float32)
        self.observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        #_, self.dt              =   vrep.simxGetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, vrep.simx_opmode_oneshot_wait)
        self.prev_linvel        =   np.zeros(3, dtype=np.float32)
        self.prev_angvel        =   np.zeros(3, dtype=np.float32)

    def _get_observation_state(self, compute_acelleration = True):
        _, position             =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, orientation          =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, lin_vel, ang_vel     =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)

        position                =   np.asarray(position, dtype=np.float32)
        orientation             =   np.asarray(orientation, dtype=np.float32)
        lin_vel, ang_vel        =   np.asarray(lin_vel, dtype=np.float32), np.asarray(ang_vel, dtype=np.float32)

        if compute_acelleration == True: lin_acel, ang_acel  =   self.compute_aceleration(lin_vel, ang_vel)
        else: lin_acel, ang_acel =   np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32) 

        position                =   position - self.targetpos
        rotation_matrix         =   GetFlatRotationMatrix(orientation)

        observation             =   OrderedDict(
                                        position=position,
                                        orientation=orientation,
                                        lin_vel=lin_vel,
                                        ang_vel=ang_vel,
                                        rotation_matrix=rotation_matrix,
                                        lin_acel=lin_acel,
                                        ang_acel=ang_acel
                                    )

        self.last_observation   =   observation

        return observation

    def compute_aceleration(self, linv, angv):
        assert linv is not None and angv is not None, "linv or angv must not be a none datatype"

        lina = (linv-self.prev_linvel)/self.dt
        anga = (angv-self.prev_angvel)/self.dt

        return lina, anga
    
    def compute_rewards(self, rowdata):
        #print('computed from child')
        pos_vector  =   rowdata['position']
        magnitud    =   np.sqrt((pos_vector * pos_vector).sum())
        return 4.0 - 1.25 * magnitud

    def compute_done(self, rowdata):
        #print('computed done from child')
        pos_vector  =   rowdata['position']
        magnitud    =   np.sqrt((pos_vector * pos_vector).sum())

        return magnitud > 3.2
    
    
    @staticmethod
    def _flat_observation_st(rowdata):
        position        =   rowdata['position']
        lin_vel         =   rowdata['lin_vel']
        ang_vel         =   rowdata['ang_vel']
        rotation_matrix =   rowdata['rotation_matrix']
        lin_acel        =   rowdata['lin_acel']
        ang_acel        =   rowdata['ang_acel']

        return np.concatenate((position, lin_vel, ang_vel, rotation_matrix, lin_acel, ang_acel))

    def _flat_observation(self, rowdata):
        return VREPQuadAccelRot._flat_observation_st(rowdata)
    
    @staticmethod
    def _get_action_space():
        action_space       =   spaces.Box(low=0.0,high=100.0,shape=(4,), dtype=np.float32)
        return action_space
    
    @staticmethod
    def _get_state_space():
        observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        return observation_space



class VREPQuadRotmat(WrapperQuad):
    def __init__(self, ip='127.0.0.1', port=19997):
        super(VREPQuadRotmat, self).__init__(ip=ip, port=port)

        self.action_space       =   spaces.Box(low=0.0,high=100.0,shape=(4,), dtype=np.float32)
        self.observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        #_, self.dt              =   vrep.simxGetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, vrep.simx_opmode_oneshot_wait)
        self.prev_linvel        =   np.zeros(3, dtype=np.float32)
        self.prev_angvel        =   np.zeros(3, dtype=np.float32)

    def _get_observation_state(self, compute_acelleration = True):
        _, position             =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, orientation          =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, lin_vel, ang_vel     =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)

        position                =   np.asarray(position, dtype=np.float32)
        orientation             =   np.asarray(orientation, dtype=np.float32)
        lin_vel, ang_vel        =   np.asarray(lin_vel, dtype=np.float32), np.asarray(ang_vel, dtype=np.float32)

        if compute_acelleration == True: lin_acel, ang_acel  =   self.compute_aceleration(lin_vel, ang_vel)
        else: lin_acel, ang_acel =   np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32) 

        position                =   position - self.targetpos
        rotation_matrix         =   GetFlatRotationMatrix(orientation)

        observation             =   OrderedDict(
                                        position=position,
                                        orientation=orientation,
                                        lin_vel=lin_vel,
                                        ang_vel=ang_vel,
                                        rotation_matrix=rotation_matrix,
                                        lin_acel=lin_acel,
                                        ang_acel=ang_acel
                                    )

        self.last_observation   =   observation

        return observation
    
    @staticmethod
    def _flat_observation_st(rowdata):
        position        =   rowdata['position']
        lin_vel         =   rowdata['lin_vel']
        ang_vel         =   rowdata['ang_vel']
        rotation_matrix =   rowdata['rotation_matrix']
        #lin_acel        =   rowdata['lin_acel']
        #ang_acel        =   rowdata['ang_acel']

        return np.concatenate((rotation_matrix, position, lin_vel, ang_vel))
        #return np.concatenate((position, lin_vel, ang_vel, rotation_matrix, lin_acel, ang_acel))

    def _flat_observation(self, rowdata):
        return VREPQuadRotmat._flat_observation_st(rowdata)

    def compute_rewards(self, rowdata):
        #print('computed from child')
        pos_vector  =   rowdata['position']
        magnitud    =   np.sqrt((pos_vector * pos_vector).sum())
        return 4.0 - 1.25 * magnitud
    
    def compute_rewards_orientation(self, rowdata):
        orientation =   rowdata['orientation']
        roll_sin    =   np.sin(orientation[0])
        pitch_sin   =   np.sin(orientation[1])
        ang_pen     =   roll_sin * roll_sin + pitch_sin * pitch_sin

        reward_angle    =   2.0 - ang_pen
        reward_distance =   self.compute_rewards_distance(rowdata)

        return reward_distance + reward_angle


    def compute_aceleration(self, linv, angv):
        assert linv is not None and angv is not None, "linv or angv must not be a none datatype"

        lina = (linv-self.prev_linvel)/self.dt
        anga = (angv-self.prev_angvel)/self.dt

        return lina, anga
    
    @staticmethod
    def _get_action_space():
        action_space       =   spaces.Box(low=0.0,high=100.0,shape=(4,), dtype=np.float32)
        return action_space
    
    @staticmethod
    def _get_state_space():
        observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        return observation_space


class VREPQuadRotmatAugment(WrapperQuad):
    def __init__(self, ip='127.0.0.1', port=19997):
        super(VREPQuadRotmatAugment, self).__init__(ip=ip, port=port)

        self.action_space       =   spaces.Box(low=0.0,high=100.0,shape=(4,), dtype=np.float32)
        self.observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        #_, self.dt              =   vrep.simxGetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, vrep.simx_opmode_oneshot_wait)
        self.prev_linvel        =   np.zeros(3, dtype=np.float32)
        self.prev_angvel        =   np.zeros(3, dtype=np.float32)

    def _get_observation_state(self, compute_acelleration = True):
        _, position             =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, orientation          =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, lin_vel, ang_vel     =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)

        position                =   np.asarray(position, dtype=np.float32)
        orientation             =   np.asarray(orientation, dtype=np.float32)
        lin_vel, ang_vel        =   np.asarray(lin_vel, dtype=np.float32), np.asarray(ang_vel, dtype=np.float32)

        if compute_acelleration == True: lin_acel, ang_acel  =   self.compute_aceleration(lin_vel, ang_vel)
        else: lin_acel, ang_acel =   np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32) 

        position                =   position - self.targetpos
        rotation_matrix         =   GetFlatRotationMatrix(orientation)

        observation             =   OrderedDict(
                                        position=position,
                                        orientation=orientation,
                                        lin_vel=lin_vel,
                                        ang_vel=ang_vel,
                                        rotation_matrix=rotation_matrix,
                                        lin_acel=lin_acel,
                                        ang_acel=ang_acel
                                    )

        self.last_observation   =   observation

        return observation
    
    @staticmethod
    def _flat_observation_st(rowdata):
        position        =   rowdata['position']
        lin_vel         =   rowdata['lin_vel']
        ang_vel         =   rowdata['ang_vel']
        rotation_matrix =   rowdata['rotation_matrix']
        orientation     =   rowdata['orientation']
        #lin_acel        =   rowdata['lin_acel']
        #ang_acel        =   rowdata['ang_acel']

        return np.concatenate((rotation_matrix, position, lin_vel, ang_vel, orientation))
        #return np.concatenate((position, lin_vel, ang_vel, rotation_matrix, lin_acel, ang_acel))

    def _flat_observation(self, rowdata):
        return VREPQuadRotmatAugment._flat_observation_st(rowdata)

    def compute_rewards_distance(self, rowdata):
        #print('computed from child')
        pos_vector  =   rowdata['position']
        magnitud    =   np.sqrt((pos_vector * pos_vector).sum())
        return 4.0 - 1.25 * magnitud
    
    def compute_rewards_ang(self, rowdata):
        orientation =   rowdata['orientation']
        roll_rad    =   orientation[0]
        pitch_rad   =   orientation[1]
        ang_pen     =   roll_rad * roll_rad + pitch_rad * pitch_rad

        reward_angle    =   2.0 - ang_pen
        reward_distance =   self.compute_rewards_distance(rowdata)

        return reward_distance + reward_angle
    
    def compute_rewards(self, rowdata):
        orientation =   rowdata['orientation']
        roll_rad    =   orientation[0]
        pitch_rad   =   orientation[1]
        ang_pen     =   roll_rad * roll_rad + pitch_rad * pitch_rad

        reward_angle    =   2.0 - ang_pen
        reward_distance =   self.compute_rewards_distance(rowdata)

        ang_vel         =   rowdata['ang_vel']
        yaw_speed       =   ang_vel[2]
        #reward_yaw_speed    =   2.0 - (yaw_speed * yaw_speed)/(100.0)
        reward_yaw_speed    =   - (yaw_speed * yaw_speed)/(200.0)
        
        return reward_distance + reward_angle + reward_yaw_speed


    def compute_aceleration(self, linv, angv):
        assert linv is not None and angv is not None, "linv or angv must not be a none datatype"

        lina = (linv-self.prev_linvel)/self.dt
        anga = (angv-self.prev_angvel)/self.dt

        return lina, anga
    
    @staticmethod
    def _get_action_space():
        action_space       =   spaces.Box(low=0.0,high=100.0,shape=(4,), dtype=np.float32)
        return action_space
    
    @staticmethod
    def _get_state_space():
        observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        return observation_space