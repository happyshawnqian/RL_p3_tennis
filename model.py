import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import ACTOR_FC1_UNITS
from hyperparameters import ACTOR_FC2_UNITS
from hyperparameters import CRITIC_FCS1_UNITS
from hyperparameters import CRITIC_FC2_UNITS

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    '''Actor (Policy) Model'''

    def __init__(self, input_dim, output_dim, seed=10, fc1_units=ACTOR_FC1_UNITS, fc2_units=ACTOR_FC2_UNITS):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # dense layers
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_dim)

        # batch norm layers
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state):
        '''Build an actor (policy) network that maps states -> actions'''

        # Reshape for BatchNorm
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))



class Critic(nn.Module):
    '''Critic Model'''

    def __init__(self, input_dim, action_size, seed=10, fcs1_units=CRITIC_FCS1_UNITS, fc2_units=CRITIC_FC2_UNITS):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(input_dim + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)

        self.fc3 = nn.Linear(fc2_units, 1)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.reset_parameters()


    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state, action):
        '''Build a critic network that maps (state, action) pairs -> Q-values'''

        # Reshape the state for BatchNorm
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        xs = torch.cat((state, action.float()), dim=1)
        x = F.relu(self.fcs1(xs))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

