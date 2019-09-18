import torch
import torch.nn as nn


class Dynamics(nn.Module):
    def __init__(self, state_shape, action_shape, stack_n=1, sthocastic=True):
        super(Dynamics, self).__init__()
        self.state_shape    =   state_shape
        self.action_shape   =   action_shape
        self.stack_n        =   stack_n
        self.output_shape   =   state_shape * (2 if sthocastic else 1)

        self.input_layer_shape  =   (state_shape + action_shape) * stack_n
        
        self.layer1         =   nn.Linear(self.input_layer_shape, 250)
        self.layer2         =   nn.Linear(250, 250)
        self.layer3         =   nn.Linear(250, 250)
        self.layer4         =   nn.Linear(250, self.output_shape)
    
    def forward(self, obs):
        x   =   torch.tanh(self.layer1(obs))
        x   =   torch.tanh(self.layer2(x))
        x   =   torch.tanh(self.layer3(x))
        x   =   torch.tanh(self.layer4(x))


        return x