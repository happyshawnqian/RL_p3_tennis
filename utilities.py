import numpy as np
import torch

# Helper functions to encode multiple agents states/actions into one
# decode encoded states/actions back to states/actions for multiple agents

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def encode(sa):
    '''
    Encode multiple agents' states or actions into one array by concatenating their
    information, thus removing (but not losing) the agent dimension in the final output.

    The output is a list intended to be inserted into a buffer memory originally not
    designed to handle multiple agents information, such as in the context of MADDPG
    Params:
        sa, list: List of Environment states or actions array, corresponding to each agent
    '''
    return np.array(sa).reshape(1, -1).squeeze()

def decode(size, num_agents, id_agent, sa, debug=False):
    '''
    Decode a batch of states or actions, which have been previously concatenated to store
    multiple agent information into one array.

    The output is a batch of Environment states or actions (torch.tensor) containing the data
    of only the agent specified.
    Param:
        size, int: size of the action space or state space to decode
        num_agent, int: Number of agents in the environment
        id_agent, int: index of the agent whose information that is going to be retracted
        sa, torch.tensor: Batch of Environment states or actions, each is one array concatenating
        the info of two agents
        debug, boolean: print debug information
    '''

    list_indices = torch.tensor([idx for idx in range(id_agent * size, id_agent * size + size)]).to(device)
    out = sa.index_select(1, list_indices)

    if (debug):
        print('\nDebug decode:\n size=', size, 'num_agents=', num_agents, 'id_agent=', id_agent, '\n')
        print('input:\n', sa, '\n output:\n', out, '\n\n\n')
    return out