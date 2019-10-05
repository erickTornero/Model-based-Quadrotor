import torch
import torch.nn as nn
import numpy as np

from IPython.core.debugger import set_trace

    
class Dynamics(nn.Module):
    def __init__(self, state_shape, action_shape, stack_n=1, sthocastic=True, actfn = torch.tanh, hlayers=(250,250,250)):
        super(Dynamics, self).__init__()
        self.sthocastic     =   sthocastic
        self.state_shape    =   state_shape[0]
        self.action_shape   =   action_shape[0]
        self.stack_n        =   stack_n
        self.output_shape   =   self.state_shape * (2 if sthocastic else 1)

        self.input_layer_shape  =   (self.state_shape + self.action_shape) * self.stack_n
        
        input_sizes         =   (self.input_layer_shape, *hlayers)
        output_sizes        =   (*hlayers, self.output_shape)
        self.layers         =   [nn.Linear(isz, osz) for isz, osz in zip(input_sizes, output_sizes)]
        self.layers         =   nn.ModuleList(self.layers)
        #self.layer1         =   nn.Linear(self.input_layer_shape, 250)
        #self.layer2         =   nn.Linear(250, 250)
        #self.layer3         =   nn.Linear(250, 250)
        #self.layer4         =   nn.Linear(250, self.output_shape)
        self.actfn          =   actfn
        self.mean_input     =   None
        self.std_input      =   None
        self.epsilon        =   None    
    
    def forward(self, obs):
        x   =   obs
        if self.actfn is not None:
            for idx in range(len(self.layers)-1):
                x   =   self.actfn(self.layers[idx](x))
        else:
            for idx in range(len(self.layers)-1):
                x   =   self.layers[idx](x)
        
        x   =   self.layers[-1](x)

        return x
    
    def predict_next_obs(self, obs, device):
        """
        Prediction of the next observation given current stack of obs.
        s_{t+1} = s_{t} + delta(next_obs)

        The observation is passed normalized, must be denormalized in order to compute the next observation
        """

        with torch.no_grad():
            if not self.sthocastic:
                x   =   self.forward(obs)
                #x   =   obs[:, self.state_shape * (self.stack_n - 1): self.state_shape * self.stack_n] + x[:, :self.state_shape]
                x   =   self.denormalize_state(obs, self.state_shape*(self.stack_n - 1), self.state_shape * self.stack_n, device) + x[:, :self.state_shape]
            else:
                pass
    
    def compute_normalization_stats(self, obs):
        self.mean_input =   np.mean(obs, axis=0)
        self.std_input  =   np.std(obs, axis=0)
        self.epsilon    =   1e-6
    
    def denormalize_state(self, obs, i_index, e_index, device):
        """ Denormalize a portion of the state """
        mean_arr        =   torch.tensor(self.mean_input[i_index:e_index], dtype=torch.float32, device=device)
        std_arr         =   torch.tensor(self.std_input[i_index:e_index],  dtype=torch.float32, device=device)

        x   =   obs[:, i_index:e_index] * (std_arr + self.epsilon) + mean_arr
        return x



class OldDynamics(nn.Module):
    def __init__(self, state_shape, action_shape, stack_n=1, sthocastic=True):
        super(OldDynamics, self).__init__()
        self.sthocastic     =   sthocastic
        self.state_shape    =   state_shape[0]
        self.action_shape   =   action_shape[0]
        self.stack_n        =   stack_n
        self.output_shape   =   self.state_shape * (2 if sthocastic else 1)

        self.input_layer_shape  =   (self.state_shape + self.action_shape) * self.stack_n
        self.layer1         =   nn.Linear(self.input_layer_shape, 250)
        self.layer2         =   nn.Linear(250, 250)
        self.layer3         =   nn.Linear(250, 250)
        self.layer4         =   nn.Linear(250, self.output_shape)

        self.mean_input     =   None
        self.std_input      =   None
        self.epsilon        =   None    
    
    def forward(self, obs):
        x   =   torch.tanh(self.layer1(obs))
        x   =   torch.tanh(self.layer2(x))
        x   =   torch.tanh(self.layer3(x))
        x   =   self.layer4(x)

        return x
    
    def predict_next_obs(self, obs, device):
        """
        Prediction of the next observation given current stack of obs.
        s_{t+1} = s_{t} + delta(next_obs)

        The observation is passed normalized, must be denormalized in order to compute the next observation
        """

        with torch.no_grad():
            if not self.sthocastic:
                x   =   self.forward(obs)
                #x   =   obs[:, self.state_shape * (self.stack_n - 1): self.state_shape * self.stack_n] + x[:, :self.state_shape]
                x   =   self.denormalize_state(obs, self.state_shape*(self.stack_n - 1), self.state_shape * self.stack_n, device) + x[:, :self.state_shape]
            else:
                pass

        return x
    
    def compute_normalization_stats(self, obs):
        self.mean_input =   np.mean(obs, axis=0)
        self.std_input  =   np.std(obs, axis=0)
        self.epsilon    =   1e-6
    
    def denormalize_state(self, obs, i_index, e_index, device):
        """ Denormalize a portion of the state """
        mean_arr        =   torch.tensor(self.mean_input[i_index:e_index], dtype=torch.float32, device=device)
        std_arr         =   torch.tensor(self.std_input[i_index:e_index],  dtype=torch.float32, device=device)

        x   =   obs[:, i_index:e_index] * (std_arr + self.epsilon) + mean_arr
        return x

@staticmethod
def OltStyleToNewStyle(olddyn, newdyn):
    for (_, value), (_, valuenew) in zip(olddyn.state_dict().items(), newdyn.state_dict().items()):
        valuenew.data.copy_(value.data)
    print("New style copied successfully")