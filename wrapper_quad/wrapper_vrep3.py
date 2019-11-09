import gym
from gym import spaces
import numpy as np
import wrapper_quad.vrep as vrep
from typing import NoReturn
import time
from random import gauss
from collections import OrderedDict


class VREPQuadSimple(gym.Env):
    def __init__(self, ip='127.0.0.1', port=19997, envname='Quadricopter', targetpos=np.zeros(3, dtype=np.float32)):
        super(VREPQuadSimple, self).__init__()
        # Initialize vrep
        self.envname            =   envname
        clientID                =   vrep.simxStart(ip, port, True, True, 5000, 0)
    
        if clientID != -1:
            print('Connection Established Successfully to IP> {} - Port> {} - ID: {}'.format(ip, port, clientID))
            self.clientID       =   clientID
            self.targetpos      =   targetpos
            _, self.dt          =   vrep.simxGetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, vrep.simx_opmode_oneshot_wait)

            self.prev_linvel    =   np.zeros(3, dtype=np.float32)
            self.prev_angvel    =   np.zeros(3, dtype=np.float32)
            #self.prev_pos
            print('Initialized with tstep>\t{}'.format(vrep.simxGetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, vrep.simx_opmode_oneshot_wait)))
        else:
            raise ConnectionError("Can't Connect with the envinronment at IP:{}, Port:{}".format(ip, port))
        
        ## Detach object target_get_random_pos_ang
        r, self.target_handler      =   vrep.simxGetObjectHandle(clientID, 'Quadricopter_target', vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectParent(clientID, self.target_handler, -1, True, vrep.simx_opmode_oneshot_wait)
        # Set signal debug:
        vrep.simxSetIntegerSignal(self.clientID, 'signal_debug', 1337, vrep.simx_opmode_oneshot)
        r, self.quad_handler         =   vrep.simxGetObjectHandle(clientID, self.envname, vrep.simx_opmode_oneshot_wait)

        print(r, self.quad_handler)
        # Define gym variables

        self.action_space       =   spaces.Box(low=0.0, high=100.0, shape=(4,), dtype=np.float32)

        self.observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # Get scripts propellers Here...!
        #self.propsignal =   ['joint' + str(i+1) for i in range(0, 4)]
        self.propsignal =   ['speedprop' + str(i+1) for i in range(0, 4)]

    def step(self, action:np.ndarray):
        """ 
            Angles en vrep go through -180º to 180, 180ª=-180º 
            Early stop for angles that commit the following rule abs(alpha) > 90º, this solve the problem of +-180º
            Compute sin & cos of Yaw angle avoid the problem of +-180ª
            orientation: [roll, pitch, sinYAW, cosYAW]    
        """
        for act, name in zip(action, self.propsignal):
            vrep.simxSetFloatSignal(self.clientID, name, act, vrep.simx_opmode_streaming)

        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)

        #position, orientation, lin_vel, ang_vel, lin_acel, ang_acel =   self._get_observation_state()
        #position, orientation, lin_vel, ang_vel =   self._get_observation_state()
        rowdata    =   self._get_observation_state()
        #rowdata         =   self._appendtuples_((rotmat, position, angvel, linvel))
        """ Roll & Pitch must be lower than 90º, else apply earlystop """
        earlystop       =   np.abs(rowdata['orientation'][:-1]) > np.pi/2.0
        """ Use sinYAW & cosYAW as features instead of YAW directly to avoid problem when YAW is near to 180º"""
        #yawangle        =   orientation[-1]
        #orientation[-1] =   np.sin(yawangle)
        #orientation     =   np.concatenate((orientation, np.array([np.cos(yawangle)])))
#
        reward          =   rowdata['position']
        distance        =   np.sqrt((reward * reward).sum())
        reward          =   4.0 -1.25 * distance
        done            =   (distance > 3.2) or (earlystop.sum() > 0.0)
        
        #rowdata         =   np.concatenate((position, orientation, lin_vel, ang_vel), axis=0)
        rowdata         =   self._flat_observation(rowdata)
         
        #done            =   earlystop.sum() > 0.0
        return (rowdata, reward, done, dict())
    
    def reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        try:
            while True:
                vrep.simxGetIntegerSignal(self.clientID, 'signal_debug', vrep.simx_opmode_blocking)
                e   =   vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
                still_running = e[1] & 1
                if not still_running:
                    break
        except: pass
        r, self.quad_handler        =   vrep.simxGetObjectHandle(self.clientID, self.envname, vrep.simx_opmode_oneshot_wait)
        r, self.target_handler      =   vrep.simxGetObjectHandle(self.clientID, 'Quadricopter_target', vrep.simx_opmode_oneshot_wait)
        # start pose
        init_position, init_ang     =   self._get_random_pos_ang(max_radius=3.1, max_angle=np.pi, respecto=self.targetpos)
        vrep.simxSetObjectPosition(self.clientID, self.quad_handler, -1, init_position, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.clientID, self.quad_handler, -1, init_ang, vrep.simx_opmode_blocking)
        ## Set target
        vrep.simxSetObjectPosition(self.clientID, self.target_handler, -1, self.targetpos, vrep.simx_opmode_oneshot)


        self.startsimulation()
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        #rdata = self._get_observation_state(False)
        #position, orientation, lin_vel, ang_vel, lin_acel, ang_acel =   self._get_observation_state(compute_acelleration=False)
        #position, orientation, lin_vel, ang_vel =   self._get_observation_state()
        rowdata =   self._get_observation_state()
        #rowdata         =   self._appendtuples_((rotmat, position, angvel, linvel))
        """ Use sinYAW & cosYAW as features instead of YAW directly to avoid problem when YAW is near to 180º"""
        #yawangle        =   orientation[-1]
        #orientation[-1] =   np.sin(yawangle)
        #orientation     =   np.concatenate((orientation, np.array([np.cos(yawangle)])), axis=0)

        #rowdata         =   np.concatenate((position, orientation, lin_vel, ang_vel, lin_acel, ang_acel), axis=0)
        #rowdata         =   np.concatenate((position, orientation, lin_vel, ang_vel), axis=0)
        rowdata = self._flat_observation(rowdata)
        return rowdata

    def render(self, close=False):
        print('Trying to render')
        # Put code if it is necessary to render
        pass

    def close(self):
        print('Exit connection from ID client> {}'.format(self.clientID))
        vrep.simxClearIntegerSignal(self.clientID, 'signal_debug', vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(2.5)
        #writer.close()
        vrep.simxFinish(-1)

    def startsimulation(self):
        if self.clientID != -1:
            vrep.simxSynchronous(self.clientID, True)
            e = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

            #self._set_boolparam(vrep.sim_boolparam_threaded_rendering_enabled, True)
            #print(e)
        else:
            raise ConnectionError('Any conection has been done')

    def _get_observation_state(self):
        _, position         =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, orientation      =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, lin_vel, ang_vel =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)
        position            =   np.asarray(position, dtype=np.float32)
        orientation         =   np.asarray(orientation, dtype=np.float32)
        lin_vel, ang_vel    =   np.asarray(lin_vel, dtype=np.float32), np.asarray(ang_vel, dtype=np.float32)
        #if compute_acelleration == True: lin_acel, ang_acel  =   self.compute_aceleration(lin_vel, ang_vel)
        #else: lin_acel, ang_acel =   np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32) 

        position            =   position - self.targetpos

        observation         =   OrderedDict(
                                    position=position,
                                    orientation=orientation,
                                    lin_vel=lin_vel,
                                    ang_vel=ang_vel
                                )
        self.last_observation   =   observation
        #return position, orientation, lin_vel, ang_vel, lin_acel, ang_acel
        #return position, orientation, lin_vel, ang_vel
        return observation

    def compute_aceleration(self, linv, angv):
        assert linv is not None and angv is not None, "linv or angv must not be a none datatype"

        lina = (linv-self.prev_linvel)/self.dt
        anga = (angv-self.prev_angvel)/self.dt

        return lina, anga
    
    def _set_boolparam(self, parameter: int, value: bool) -> NoReturn:
        """Sets boolean parameter of V-REP simulation.
        Args:
            parameter: Parameter to be set.
            value: Boolean value to be set.
        """
        res = vrep.simxSetBooleanParameter(self.clientID, parameter, value,
                                           vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag), (
            'Could not set boolean parameter!')
            
    def _get_boolparam(self, parameter: int) -> bool:
        res, value = vrep.simxGetBooleanParameter(self.clientID, parameter,
                                                  vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag), (
            'Could not get boolean parameter!')
        return value
    
    def _clear_gui(self) -> NoReturn:
        """Clears GUI with unnecessary elements like model hierarchy, library browser and
        console. Also this method enables threaded rendering.
        """
        self._set_boolparam(vrep.sim_boolparam_hierarchy_visible, False)
        self._set_boolparam(vrep.sim_boolparam_console_visible, False)
        self._set_boolparam(vrep.sim_boolparam_browser_visible, False)

    def _getGaussVectorOrientation(self): 
        x = [gauss(0, 0.6) for _ in range(3)]
        return  np.asarray(x, dtype=np.float32)

    def _get_random_pos_ang(self, max_radius = 3.2, max_angle = np.pi, respecto:np.ndarray=None):
        if respecto is None:
            respecto    =   np.zeros(3, dtype=np.float32)

        max_radius_per_axis =   np.sqrt(max_radius * max_radius / 3.0)
        sampledpos          =   np.random.uniform(-max_radius_per_axis, max_radius_per_axis, 3) + respecto
        sampledangle        =   self._getGaussVectorOrientation()

        return sampledpos, sampledangle

    @staticmethod
    def _flat_observation_st(rowdata):
        position        =   rowdata['position']
        orientation     =   rowdata['orientation']
        lin_vel         =   rowdata['lin_vel']
        ang_vel         =   rowdata['ang_vel']

        yawangle        =   orientation[-1]
        orientation[-1] =   np.sin(yawangle)
        orientation     =   np.concatenate((orientation, np.array([np.cos(yawangle)])), axis=0)


        return np.concatenate((position, orientation, lin_vel, ang_vel))

    def _flat_observation(self, rowdata):
        return VREPQuadSimple._flat_observation_st(rowdata)

    @staticmethod
    def _get_action_space():
        action_space       =   spaces.Box(low=0.0,high=100.0,shape=(4,), dtype=np.float32)
        return action_space
    
    @staticmethod
    def _get_state_space():
        observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        return observation_space