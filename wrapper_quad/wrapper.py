# This will envolve the environment VREP quadrotor by an openai gym
# Command to run the vrep:
# ./vrep -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE
import gym
from gym import spaces
import numpy as np

from .utility import GetFlatRotationMatrix
# From environment
import sys
#sys.path.append('.')
import wrapper_quad.vrep as vrep
from typing import NoReturn
import time
from random import gauss
import numpy as np
from collections import OrderedDict

# if debug
#from tensorboardX import SummaryWriter
#writer  =   SummaryWriter()

# Environment to pass the target position just on the initial state
class WrapperQuad(gym.Env):

    def __init__(self, ip='127.0.0.1', port=19997, envname='Quadricopter', targetpos=np.zeros(3, dtype=np.float32)):
        super(WrapperQuad, self).__init__()
        # Initialize vrep
        self.envname            =   envname
        self.target_name        =   'Quadricopter_target'
        #vrep.simxFinish(-1)
        clientID                =   vrep.simxStart(ip, port, True, True, 5000, 0)
        if clientID != -1:
            print('Connection Established Successfully to IP> {} - Port> {} - ID: {}'.format(ip, port, clientID))
            self.clientID       =   clientID
            self.targetpos      =   targetpos
            _, self.dt              =   vrep.simxGetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, vrep.simx_opmode_oneshot_wait)
            #self.prev_pos
            print('Initialized with tstep>\t{} seconds'.format(self.dt))
        else:
            raise ConnectionError("Can't Connect with the envinronment at IP:{}, Port:{}".format(ip, port))
        
        #pass

        if not self._get_boolparam(vrep.sim_boolparam_headless):
            self._clear_gui()

        ## Detach object target_get_random_pos_ang
        r, self.target_handler      =   vrep.simxGetObjectHandle(clientID, 'Quadricopter_target', vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectParent(clientID, self.target_handler, -1, True, vrep.simx_opmode_oneshot_wait)
        # Set signal debug:
        vrep.simxSetIntegerSignal(self.clientID, 'signal_debug', 1337, vrep.simx_opmode_oneshot)
        r, self.quad_handler         =   vrep.simxGetObjectHandle(clientID, self.envname, vrep.simx_opmode_oneshot_wait)

        print(r, self.quad_handler)
        # Define gym variables
        """ These properties must be overloaded """
        self.action_space       =   spaces.Box(low=0.0, high=100.0, shape=(4,), dtype=np.float32)

        self.observation_space  =   spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        self.last_observation   =   None

        # Get scripts propellers Here...!
        #self.propsignal =   ['joint' + str(i+1) for i in range(0, 4)]
        self.propsignal =   ['speedprop' + str(i+1) for i in range(0, 4)]

    def step(self, action:np.ndarray):
        for act, name in zip(action, self.propsignal):
            vrep.simxSetFloatSignal(self.clientID, name, act, vrep.simx_opmode_streaming)
        
        #vrep.simxSetFloatSignal(self.clientID, self.propsignal1)
        # sincronyze
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        
        #rotmat, position, angvel, linvel =   self._get_observation_state()
        """ _get_observation_state() must be overloaded! """
        rowdata         =   self._get_observation_state()
        #rowdata         =   self._appendtuples_((rotmat, position, angvel, linvel))

        """ compute_rewards(rowdata) must be overloaded"""
        reward          =   self.compute_rewards(rowdata)

        """ compute_done(rowdata) must be overloaded """
        done             =   self.compute_done(rowdata)

        """ Flatten data into numpy vector""" 
        rowdata     =   self._flat_observation(rowdata)

        return (rowdata, reward, done, dict())

        # Compute The reward function

    def reset(self):
        #print('Reset -ing id> ', self.clientID)
        # Put code when reset here
        #r = vrep.simxSetObjectPosition(self.clientID, self.quad_handler, -1, np.array([0.0,0.0,0.5]), vrep.simx_opmode_oneshot_wait)
        #r = vrep.simxCallScriptFunction(self.clientID, 'Quadricopter_target', vrep.sim_scripttype_childscript, 'sysCall_custom_reset', np.array([]), np.array([]), np.array([]), bytearray(), vrep.simx_opmode_blocking)
        # pass
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        try:
            while True:       
                vrep.simxGetIntegerSignal(self.clientID, 'signal_debug', vrep.simx_opmode_blocking)
                e   =   vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
                still_running = e[1] & 1
                if not still_running:
                    break
        except: pass
        
        #print('Totally stopped> ID> ', self.clientID)

        # Reset quadrotor
        r, self.quad_handler        =   vrep.simxGetObjectHandle(self.clientID, self.envname, vrep.simx_opmode_oneshot_wait)
        r, self.target_handler      =   vrep.simxGetObjectHandle(self.clientID, 'Quadricopter_target', vrep.simx_opmode_oneshot_wait)
        # start posedistance        =   np.sqrt((reward * reward).sum())
        #init_position, init_ang     =   self._get_random_pos_ang(max_radius=3.1, max_angle=np.pi, respecto=self.targetpos)
        #vrep.simxSetObjectPosition(self.clientID, self.quad_handler, -1, init_position, vrep.simx_opmode_blocking)
        #vrep.simxSetObjectOrientation(self.clientID, self.quad_handler, -1, init_ang, vrep.simx_opmode_blocking)
        self.set_states()
        ## Set target
        vrep.simxSetObjectPosition(self.clientID, self.target_handler, -1, self.targetpos, vrep.simx_opmode_oneshot)

        self.startsimulation()

        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)

        rowdata = self._get_observation_state()
        return self._flat_observation(rowdata)

    def render(self, close=False):
        print('Trying to render')
        # Put code if it is necessary to render
        pass

    def startsimulation(self):
        if self.clientID != -1:
            self._set_floatparam(vrep.sim_floatparam_simulation_time_step, 0.01)
            vrep.simxSynchronous(self.clientID, True)
            e = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

            #self._set_boolparam(vrep.sim_boolparam_threaded_rendering_enabled, True)
            #print(e)
        else:
            raise ConnectionError('Any conection has been done')
    def __del__(self):
        print('Remove instant')
        self.close()

    def _set_floatparam(self, parameter: int, value: float) ->NoReturn:
        res =   vrep.simxSetFloatingParameter(self.clientID, parameter, value, vrep.simx_opmode_oneshot)
        #print(res)
        assert (res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag), ('Could not set float parameters!')

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

    def _clear_gui(self) -> NoReturn:
        """Clears GUI with unnecessary elements like model hierarchy, library browser and
        console. Also this method enables threaded rendering.
        """
        self._set_boolparam(vrep.sim_boolparam_hierarchy_visible, False)
        self._set_boolparam(vrep.sim_boolparam_console_visible, False)
        self._set_boolparam(vrep.sim_boolparam_browser_visible, False)

    def _get_boolparam(self, parameter: int) -> bool:
        res, value = vrep.simxGetBooleanParameter(self.clientID, parameter,
                                                  vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag), (
            'Could not get boolean parameter!')
        return value
    
    def _appendtuples_(self, xdat):
        x   =   np.empty(0, dtype=np.float32)
        for dt in xdat:
            x   =   np.append(x, dt, axis=0)

        return x
    
    def _getGaussVectorOrientation(self): 
        x   =   [gauss(0, 0.6) for _ in range(3)]
        x   =   np.clip(x, -np.pi/2.0+0.001, np.pi/2.0-0.001)
        return  np.asarray(x, dtype=np.float32)

    def _get_random_pos_ang(self, max_radius = 3.2, max_angle = np.pi, respecto:np.ndarray=None):
        if respecto is None:
            respecto    =   np.zeros(3, dtype=np.float32)

        max_radius_per_axis =   np.sqrt(max_radius * max_radius / 3.0)
        sampledpos          =   np.random.uniform(-max_radius_per_axis, max_radius_per_axis, 3) + respecto
        sampledangle        =   self._getGaussVectorOrientation()

        return sampledpos, sampledangle

    
    def close(self):
        print('Exit connection from ID client> {}'.format(self.clientID))
        vrep.simxClearIntegerSignal(self.clientID, 'signal_debug', vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(2.5)
        #writer.close()
        vrep.simxFinish(-1)
    
    def set_states(self, pos=None, ang=None):
        """
            Stablish the position, and angular condition
        """
        if pos == None or ang == None:
            init_position, init_ang     =   self._get_random_pos_ang(max_radius=3.1, max_angle=np.pi, respecto=self.targetpos)
            if pos == None: pos=init_position
            if ang == None: ang=init_ang
        vrep.simxSetObjectPosition(self.clientID, self.quad_handler, -1, pos, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.clientID, self.quad_handler, -1, ang, vrep.simx_opmode_blocking)

    def compute_rewards(self, data):
        position    =   data['position']
        position    =   np.sqrt((position * position).sum())
        return 4.0 - 1.25 * position

    def compute_done(self, data):
        position    =   data['position']
        position    =   np.sqrt((position * position).sum())
        return position > 3.2

    def _get_observation_state(self):
        
        _, position         =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, orientation      =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        _, lin_vel, ang_vel =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)
        # Flat and join states!
        RotMat          =   GetFlatRotationMatrix(orientation)
        
        """ Test relative position """
        position        =   position  - self.targetpos
        

        observation     =   OrderedDict(
                                position=position,
                                orientation=orientation,
                                lin_vel=lin_vel,
                                ang_vel=ang_vel,
                                rotation_matrix=RotMat
                            )
        
        self.last_observation   =   observation

        return observation

    def _flat_observation(self, obs):
        #return np.concatenate(tuple(obs.values()))
        pos         =   obs['position']
        lin_vel     =   obs['lin_vel']
        ang_vel     =   obs['ang_vel']
        rot_mat     =   obs['rotation_matrix']
        return  np.concatenate((rot_mat, pos, ang_vel, lin_vel))

    def _get_last_observation(self):
        return self.last_observation