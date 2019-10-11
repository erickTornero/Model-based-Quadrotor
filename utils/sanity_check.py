from mbrl.network import Dynamics
from mbrl.mpc import RandomShooter
from rolls import rollouts
from mbrl.runner import StackStAct
import torch
class SanityCheck:
    """
        Sanity Check class
        ===================
        Provides an interface to check how well the dynamics is predicting
        Given an initial state S0, taken at point "t_init" time-step, predict the
        following "horizon" future timesteps.

        A comparison is provided between the states generated by the artificial dynamics
        and the ground truth dynamics. 

        First, actions are generated by the MPC and dynamics, the resulting true-actions
        and true-states are stored.
        Then, the same actions is taken from the initial state to generate a new set of states
        these new states are stored as artificial-states, the a comparison is performed.

        @Parameters:
        h               :   horizon
        c               :   number of candidates to the MPC
        mpc             :   Model Predictive Controler
        env             :   Environment, 'QuadrotorEnv'
        t_init          :   t_step to take the initial State
        traj            :   Trajectory over will be generated the states
        max_path_length :   The maximum length of a path
    """

    def __init__(self, h, c, dynamics:Dynamics, mpc, env, t_init, traj, max_path_length=250):
        self.horizon            =   h
        self.candidates         =   c
        self.dynamics           =   dynamics
        self.mpc                =   mpc
        self.env                =   env
        self.t_init             =   t_init
        self.trajectory         =   traj
        self.max_path_length    =   max_path_length
        self.nstack             =   dynamics.stack_n

    def get_state_actions(self):
        """ Generate one rollout """
        path    =   rollouts(self.dynamics, self.env, self.mpc, 1, self.max_path_length, None, self.trajectory)
        gt_states   =   path[0]['observations'][self.t_init:, 18*(self.nstack-1):]
        gt_actions  =   path[0]['actions'][self.t_init:,:]

        init_stackobs    =  gt_states[0].reshape(self.nstack, -1)
        init_stackacts   =  gt_actions[0].reshape(self.nstack, -1)
           
        stack_as = StackStAct(self.env.action_space.shape, self.env.observation_space.shape, n=self.nstack)

        stack_as.fill_with_stack(init_stackobs, init_stackacts)

        device      =   self.dynamics.device
        art_states  =   [gt_states[0]]
        art_actions =   [gt_actions[0]]
        for i in range(1, self.horizon+1):
            obs         =   stack_as.get()
            obs         =   self.mpc.normalize_(obs)
            obs_tensor  =   torch.tensor(obs, dtype=torch.float32, device=device)
            obs_tensor.unsqueeze_(0)
            state_gen   =   self.dynamics.predict_next_obs(obs_tensor)

