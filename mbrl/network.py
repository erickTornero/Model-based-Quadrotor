import torch
import torch.nn as nn

from IPython.core.debugger import set_trace

class Dynamics(nn.Module):
    def __init__(self, state_shape, action_shape, stack_n=1, sthocastic=True):
        super(Dynamics, self).__init__()
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
    
    def forward(self, obs):
        x   =   torch.tanh(self.layer1(obs))
        x   =   torch.tanh(self.layer2(x))
        x   =   torch.tanh(self.layer3(x))
        x   =   self.layer4(x)

        return x
    
    def predict_next_obs(self, obs):
        """
        Prediction of the next observation given current stack of obs.
        s_{t+1} = s_{t} + delta(next_obs)
        """

        with torch.no_grad():
            if not self.sthocastic:
                x   =   self.forward(obs)
                x   =   obs[:, self.state_shape * (self.stack_n - 1): self.state_shape * self.stack_n] + x[:, :self.state_shape]
            else:
                pass

        return x