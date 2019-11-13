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

# if debug
#from tensorboardX import SummaryWriter
#writer  =   SummaryWriter()

# Environment to pass the target position just on the initial state
class VREPQuad(gym.Env):

    def __init__(self, ip='127.0.0.1', port=19997, envname='Quadricopter', targetpos=np.zeros(3, dtype=np.float32), maxdist = 300.0):
        super(VREPQuad, self).__init__()
        # Initialize vrep
        self.envname            =   envname
        #vrep.simxFinish(-1)
        clientID                =   vrep.simxStart(ip, port, True, True, 5000, 0)
        if clientID != -1:
            print('Connection Established Successfully to IP> {} - Port> {} - ID: {}'.format(ip, port, clientID))
            self.clientID       =   clientID
            self.targetpos      =   targetpos
            self.max_distance   =   maxdist   
            self.counterclose   =   0
            self.distance_close =   0.01
            self.maxncounter    =   20
            self.timestep       =   0
            self.cumulative_rw  =   0.0
            self.episod         =   0
            #self.prev_pos
            print('Initialized with tstep>\t{}'.format(vrep.simxGetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, vrep.simx_opmode_oneshot_wait)))
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

        self.action_space       =   spaces.Box(low=0.0, high=100.0, shape=(4,), dtype=np.float32)

        self.observation_space  =   spaces.Box(low=-1000.0, high=1000.0, shape=(18,), dtype=np.float32)

        # Get scripts propellers Here...!
        #self.propsignal =   ['joint' + str(i+1) for i in range(0, 4)]
        self.propsignal =   ['speedprop' + str(i+1) for i in range(0, 4)]
        
        #self.propsignal2 = 'speedprop2'
        #self.propsignal3 = 'speedprop3'
        #self.propsignal4 = 'speedprop4'

    def step(self, action:np.ndarray):
        # assume of action be an np.array of dimension (4,)
        # Act!
        #action  =   action + 50.0
        #print('{}-act> {}'.format(self.clientID, action))
        #action = np.squeeze(action, 0)
        for act, name in zip(action, self.propsignal):
            vrep.simxSetFloatSignal(self.clientID, name, act, vrep.simx_opmode_streaming)
        
        #print(action)
        self.timestep   =   self.timestep + 1
        #vrep.simxSetFloatSignal(self.clientID, self.propsignal1)
        # sincronyze
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        
        # Put code here:

        ## Do an action to environment

        ## Get states
        rotmat, position, angvel, linvel =   self._get_observation_state()
        #print(rotmat, position, angvel, linvel)

        rowdata         =   self._appendtuples_((rotmat, position, angvel, linvel))
        #_, position        =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        #orientation     =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        #velocity        =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)
        #quaternion      =   vrep.simxGetObjectQuaternion(self.clientID,  self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
#
        ## Flat and join states!
        #RotMat          =   GetFlatRotationMatrix(orientation[1])
        #rowdata         =   np.append(RotMat, position)
        #rowdata         =   np.append(rowdata, velocity[1])
        #rowdata         =   np.append(rowdata, velocity[2])

        convergence_dist = position-self.prev_pos
        self.prev_pos   =   position
        diff_close      =   np.sqrt((convergence_dist * convergence_dist).sum())
        self.counterclose = (self.counterclose + 1) if diff_close < self.distance_close else 0

        #reward          =   self.targetpos - position
        reward          =   position
        distance        =   np.sqrt((reward * reward).sum())
        reward          =   4.0 -1.25 * distance

        #TODO: Temporal reward penalization, must be abstracted more
        #rollpitch_  =   linvel[:2]/10.0
        #yawvel      =   linvel[2]/30.0
        #angvel_penalization = np.sqrt((rollpitch_*rollpitch_).sum()+yawvel*yawvel)
        #reward = reward - angvel_penalization 
        #------------Reward version 2------------#
        #roll_rad    =   np.arcsin(-rotmat[3])
        #cosroll     =   np.cos(roll_rad)
        #pitch_rad   =   np.arcsin(rotmat[7])/cosroll
        #yaw_rad     =   np.arcsin(rotmat[6])/cosroll

        # EndTODO:
        #if distance > 3.2:
        #    reward = 0.0
        self.cumulative_rw  +=  reward

        #print(reward)
        
        #print(action)
        #done            =   (True if self.timestep >= 250 else False) | (distance > 3.2)
        done             =   (distance > 3.2)
        #done            =   False
        #done            =   distance > self.max_distance or self.counterclose > self.maxncounter

        if done:
            self.episod += 1
            if self.episod % 10 == 0:
                pass
                #print('From Target> {:04.2f}'.format(distance))
                #print('Episode:>\t{}\tFrom Target> {:04.2f}\ttimesteps> {}\t Ep Reward> {:05.2f}'.format(self.episod, distance, self.timestep, self.cumulative_rw))
                #print('From target: {}'.format(distance))
                #print('timesteps> ',self.timestep)
                #print('Cum_rw>:', self.cumulative_rw)
                #writer.add_scalar('data/eprw', self.cumulative_rw, self.episod)
                #writer.add_scalar('data/dist_from_targ', distance, self.episod)
        return (rowdata, reward, done, dict())

        # Compute The reward function

    def reset(self):
        #print('Reset -ing id> ', self.clientID)
        # Put code when reset here
        #r = vrep.simxSetObjectPosition(self.clientID, self.quad_handler, -1, np.array([0.0,0.0,0.5]), vrep.simx_opmode_oneshot_wait)
        #r = vrep.simxCallScriptFunction(self.clientID, 'Quadricopter_target', vrep.sim_scripttype_childscript, 'sysCall_custom_reset', np.array([]), np.array([]), np.array([]), bytearray(), vrep.simx_opmode_blocking)
        # pass
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        ## while True:
        ##     e = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
        ##     still_running = e[1] & 1
        ##     print(e)
        ##     if not still_running:
        ##         break
        ##
        #time.sleep(3.0)
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
        # start pose
        init_position, init_ang     =   self._get_random_pos_ang(max_radius=3.1, max_angle=np.pi, respecto=self.targetpos)
        vrep.simxSetObjectPosition(self.clientID, self.quad_handler, -1, init_position, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.clientID, self.quad_handler, -1, init_ang, vrep.simx_opmode_blocking)
        ## Set target
        vrep.simxSetObjectPosition(self.clientID, self.target_handler, -1, self.targetpos, vrep.simx_opmode_oneshot)
        #print('PUT>\t',init_ang)
        # Start simulation
        #print('Starting simulation')
        #vrep.simxSynchronousTrigger(self.clientID)
        #vrep.simxGetPingTime(self.clientID)
        self.startsimulation()
        #vrep.simxSynchronous(self.clientID, True)
        #e = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        #print(e)
        # get observation

        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        #vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
#       
        
        #vrep.simxSynchronous(self.clientID, True)
        #vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        #print('s')
        rdata = self._get_observation_state()
        #orientation     =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        #print('GET>\t', orientation)
        self.prev_pos = np.asarray(rdata[1])
        # Reset parameters
        self.counterclose   =   0
        #self.timestep       =   0
        self.cumulative_rw  =   0.0
        #print('Finish reset in ID> ', self.clientID)
        return self._appendtuples_(rdata)

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
    
    def _get_observation_state(self):
        _, position     =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        orientation     =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        velocity        =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)
        quaternion      =   vrep.simxGetObjectQuaternion(self.clientID,  self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)

        # Flat and join states!
        RotMat          =   GetFlatRotationMatrix(orientation[1])
        #rowdata         =   np.append(RotMat, position)
        #rowdata         =   np.append(rowdata, velocity[1])
        #rowdata         =   np.append(rowdata, velocity[2])
        
        """ Test relative position """
        position        =   position  - self.targetpos
        #print(position)


        return (RotMat, np.asarray(position), np.asarray(velocity[1]), np.asarray(velocity[2]))        

    #def _appendtuples_(self, rotmat, pos, angvel, linvel):
    def _appendtuples_(self, xdat):
        x   =   np.empty(0, dtype=np.float32)
        for dt in xdat:
            x   =   np.append(x, dt, axis=0)

        return x
    
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

    
    def close(self):
        print('Exit connection from ID client> {}'.format(self.clientID))
        vrep.simxClearIntegerSignal(self.clientID, 'signal_debug', vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(2.5)
        #writer.close()
        vrep.simxFinish(-1)
    
    @property
    def states(self):
        return dict(shape=tuple(self.observation_space.shape), type='float')

    @property
    def actions(self):
        return dict(type='float', shape=self.action_space.low.shape, min_value=np.float(self.action_space.low[0]), max_value=np.float(self.action_space.high[0]))
    
    def execute(self, action):
        st, rw, done, _ = self.step(action)
        return st, done, rw
    
    
## Test
def TestEnv():
    env = VREPQuad(ip='192.168.0.36', port=19999)

    for ep in range(10):
        env.reset()
        done = False
        cum_rw = 0.0
        while not done:
            act_ = np.random.uniform(6.0, 8.0, 4)
            ob, rw, done, info = env.step(act_)
            #print(rw)
            cum_rw = cum_rw + rw

        print('Reward>', cum_rw)
    
    env.close()


#TestEnv()
#vrepX = VREPQuad(ip='192.168.0.36',port=19999)
#
#import time
#time.sleep(1)
#
#ob, rw =    vrepX.step(np.array([4.4, 4, 4.3, 4])) 
#print('observation> ', ob)
#print('reward>', rw)
